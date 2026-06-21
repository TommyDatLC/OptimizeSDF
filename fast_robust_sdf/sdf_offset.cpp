#include "sdf_offset.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <numeric>

using Clock = std::chrono::high_resolution_clock;

// ============================================================
// Utility: log with tag
// ============================================================
static void logStep(const std::string& tag, const std::string& msg) {
    std::cout << "[" << tag << "] " << msg << "\n";
}

// ============================================================
// AABB::intersect  (slab method -- same algorithm as OptiX BVH)
// ============================================================
bool AABB::intersect(const Vec3& ro, const Vec3& rd, double& tmin, double& tmax) const
{
    tmin = -1e18;
    tmax =  1e18;
    for (int i = 0; i < 3; ++i) {
        double o = (i == 0 ? ro.x : i == 1 ? ro.y : ro.z);
        double d = (i == 0 ? rd.x : i == 1 ? rd.y : rd.z);
        double lo = (i == 0 ? bmin.x : i == 1 ? bmin.y : bmin.z);
        double hi = (i == 0 ? bmax.x : i == 1 ? bmax.y : bmax.z);

        if (std::abs(d) < 1e-12) {
            if (o < lo || o > hi) return false;
        } else {
            double inv = 1.0 / d;
            double t0 = (lo - o) * inv;
            double t1 = (hi - o) * inv;
            if (t0 > t1) std::swap(t0, t1);
            tmin = std::max(tmin, t0);
            tmax = std::min(tmax, t1);
            if (tmin > tmax) return false;
        }
    }
    return true;
}

// ============================================================
// LoadObj -- reuses OBJ parsing pattern from Model::ReadFromObjFile
// ============================================================
void SdfOffset::loadObj(const std::string& filename)
{
    auto t0 = Clock::now();
    logStep("LOAD", "Loading OBJ: " + filename);

    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    vertices.clear();
    triangles.clear();
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            double x, y, z;
            iss >> x >> y >> z;
            vertices.push_back({x, y, z});
        } else if (prefix == "f") {
            std::array<size_t, 3> idx;
            std::string tok;
            for (int i = 0; i < 3; ++i) {
                iss >> tok;
                size_t pos = tok.find('/');
                if (pos != std::string::npos) tok = tok.substr(0, pos);
                idx[i] = std::stoull(tok) - 1;  // OBJ is 1-based
            }
            triangles.push_back({idx[0], idx[1], idx[2]});
        }
    }

    timeLoad = std::chrono::duration<double>(Clock::now() - t0).count();
    logStep("LOAD", "Vertices: " + std::to_string(vertices.size()) +
            "  Triangles: " + std::to_string(triangles.size()) +
            "  Time: " + std::to_string(timeLoad) + " s");
}

// ============================================================
// ComputeVertexNormals -- reuses cross-product pattern from
//   ModelHelper.cu GPUNormalCaculation + GPUNormalizeVertexNormal
// ============================================================
void SdfOffset::computeVertexNormals()
{
    auto t0 = Clock::now();
    logStep("NORMALS", "Computing vertex normals (area-weighted accumulation)...");

    vertexNormals.assign(vertices.size(), {0, 0, 0});

    // Accumulate face normals weighted by triangle area
    //   (same logic as GPUNormalCaculation: cross two edges, atomicAdd)
    for (auto& tri : triangles) {
        Vec3 v0 = vertices[tri.i0], v1 = vertices[tri.i1], v2 = vertices[tri.i2];
        Vec3 e1 = v1 - v0;
        Vec3 e2 = v2 - v0;
        Vec3 fn = e1.cross(e2);          // length = 2 * area, direction = face normal

        // Accumulate to all three vertices (same as atomicAdd in GPU kernel)
        vertexNormals[tri.i0] += fn;
        vertexNormals[tri.i1] += fn;
        vertexNormals[tri.i2] += fn;
    }

    // Normalize  (same as GPUNormalizeVertexNormal kernel)
    for (auto& n : vertexNormals) {
        double len = n.length();
        if (len > 1e-12)
            n = n / len;
        else
            n = {0, 0, 1};   // fallback for degenerate normals
    }

    timeNormals = std::chrono::duration<double>(Clock::now() - t0).count();
    logStep("NORMALS", "Done  Time: " + std::to_string(timeNormals) + " s");
}

// ============================================================
// BuildOffsetComponents -- paper Section 4
//   For each triangle: 3 spheres + 3 cylinders + 1 slab
// ============================================================
void SdfOffset::computeComponentObb_(OffsetComponent& c)
{
    switch (c.type) {
        case OffsetComponentType::Sphere:
            c.obb.bmin = c.obb.bmax = c.center;
            c.obb.expandPad(c.center, c.radius);
            break;

        case OffsetComponentType::Cylinder: {
            c.obb.bmin = c.obb.bmax = c.cylA;
            c.obb.expand(c.cylB);
            c.obb.expandPad(c.cylA, c.cylRadius);
            c.obb.expandPad(c.cylB, c.cylRadius);
            break;
        }
        case OffsetComponentType::Slab: {
            Vec3 pts[6];
            for (int i = 0; i < 3; ++i) {
                Vec3 p = (i == 0 ? c.slabOrigin : i == 1 ? c.slabOrigin + c.slabU : c.slabOrigin + c.slabV);
                pts[i]   = p + c.slabNormal * (c.slabThickness * 0.5);
                pts[i+3] = p - c.slabNormal * (c.slabThickness * 0.5);
            }
            c.obb.bmin = c.obb.bmax = pts[0];
            for (int i = 1; i < 6; ++i) c.obb.expand(pts[i]);
            break;
        }
    }
}

void SdfOffset::buildOffsetComponents()
{
    auto t0 = Clock::now();
    logStep("OFFSET", "Building offset surface components...");

    // Compute offset radius as fraction of bounding-box diagonal (paper: 5%)
    Vec3 gmin{ 1e18, 1e18, 1e18}, gmax{-1e18, -1e18, -1e18};
    for (auto& v : vertices) { gmin.x = std::min(gmin.x, v.x); gmax.x = std::max(gmax.x, v.x);
                                gmin.y = std::min(gmin.y, v.y); gmax.y = std::max(gmax.y, v.y);
                                gmin.z = std::min(gmin.z, v.z); gmax.z = std::max(gmax.z, v.z); }
    double bboxDiag = (gmax - gmin).length();
    offsetRadius = bboxDiag * 0.05;
    logStep("OFFSET", "BB diagonal: " + std::to_string(bboxDiag) +
            "  offset radius (5%): " + std::to_string(offsetRadius));

    // Reserve: 3N spheres + 3N cylinders + N slabs = 7N components
    size_t N = triangles.size();
    components_.clear();
    components_.reserve(7 * N);

    AABB globalBounds;
    globalBounds.bmin = gmin;
    globalBounds.bmax = gmax;

    for (size_t fi = 0; fi < N; ++fi) {
        auto& tri = triangles[fi];
        Vec3 verts[3] = {vertices[tri.i0], vertices[tri.i1], vertices[tri.i2]};

        // 3 spheres at vertices
        for (int i = 0; i < 3; ++i) {
            OffsetComponent sc;
            sc.type   = OffsetComponentType::Sphere;
            sc.center = verts[i];
            sc.radius = offsetRadius;
            computeComponentObb_(sc);
            components_.push_back(sc);
        }

        // 3 cylinders along edges
        for (int i = 0; i < 3; ++i) {
            Vec3 a = verts[i], b = verts[(i + 1) % 3];
            Vec3 d = b - a;
            double len = d.length();
            if (len < 1e-12) continue;

            OffsetComponent cc;
            cc.type       = OffsetComponentType::Cylinder;
            cc.cylA       = a;
            cc.cylB       = b;
            cc.cylDir     = d / len;
            cc.cylLen     = len;
            cc.cylRadius  = offsetRadius;
            computeComponentObb_(cc);
            components_.push_back(cc);
        }

        // 1 slab (tri-prism) for the face
        {
            Vec3 v0 = vertices[tri.i0], v1 = vertices[tri.i1], v2 = vertices[tri.i2];
            Vec3 e1 = v1 - v0;
            Vec3 e2 = v2 - v0;
            Vec3 n  = e1.cross(e2).normalized();

            OffsetComponent sl;
            sl.type          = OffsetComponentType::Slab;
            sl.slabNormal    = n;
            sl.slabU         = e1;
            sl.slabV         = e2;
            sl.slabOrigin    = v0;
            sl.slabThickness = 2.0 * offsetRadius;
            computeComponentObb_(sl);
            components_.push_back(sl);
        }
    }

    timeOffset = std::chrono::duration<double>(Clock::now() - t0).count();
    logStep("OFFSET", "Components: " + std::to_string(components_.size()) +
            "  Time: " + std::to_string(timeOffset) + " s");
}

// ============================================================
// BuildObbTree -- top-down BVH (same pattern as OptiX BVH build)
// ============================================================
void SdfOffset::buildObbTreeRecursive_(std::vector<OffsetComponent*>& refs,
                                       int begin, int end, int depth,
                                       std::unique_ptr<OBBNode>& node)
{
    node = std::make_unique<OBBNode>();

    // Compute bounds of all elements in [begin, end)
    node->bounds = refs[begin]->obb;
    for (int i = begin + 1; i < end; ++i)
        node->bounds.merge(refs[i]->obb);

    const int LEAF_THRESHOLD = 8;
    if (end - begin <= LEAF_THRESHOLD) {
        node->isLeaf  = true;
        node->firstIdx = begin;
        node->count    = end - begin;
        return;
    }

    // Split along the longest axis of the bounding box centroid spread
    Vec3 centroidMin{ 1e18,  1e18,  1e18};
    Vec3 centroidMax{-1e18, -1e18, -1e18};
    for (int i = begin; i < end; ++i) {
        Vec3 c = refs[i]->obb.center();
        centroidMin.x = std::min(centroidMin.x, c.x);
        centroidMin.y = std::min(centroidMin.y, c.y);
        centroidMin.z = std::min(centroidMin.z, c.z);
        centroidMax.x = std::max(centroidMax.x, c.x);
        centroidMax.y = std::max(centroidMax.y, c.y);
        centroidMax.z = std::max(centroidMax.z, c.z);
    }

    Vec3 diff = centroidMax - centroidMin;
    int axis = 0;
    if (diff.y > diff.x && diff.y > diff.z) axis = 1;
    else if (diff.z > diff.x && diff.z > diff.y) axis = 2;

    double splitVal = (axis == 0 ? centroidMin.x + diff.x * 0.5
                     : axis == 1 ? centroidMin.y + diff.y * 0.5
                                 : centroidMin.z + diff.z * 0.5);

    auto pred = [&](const OffsetComponent* a) {
        double ca = (axis == 0 ? a->obb.center().x : axis == 1 ? a->obb.center().y : a->obb.center().z);
        return ca < splitVal;
    };

    auto mid = std::partition(refs.begin() + begin, refs.begin() + end, pred);
    int m = static_cast<int>(mid - refs.begin());
    if (m == begin || m == end) m = begin + (end - begin) / 2;  // fallback

    buildObbTreeRecursive_(refs, begin, m, depth + 1, node->left);
    buildObbTreeRecursive_(refs, m,     end, depth + 1, node->right);
}

void SdfOffset::buildObbTree()
{
    auto t0 = Clock::now();
    logStep("OBB_TREE", "Building OBB tree over " + std::to_string(components_.size()) + " components...");

    // Build a pointer array into components_ and partition it in tree order.
    // The tree's leaves store (firstIdx, count) indexing into componentRefs_.
    // componentRefs_ must stay alive as long as the tree is used.

    std::vector<OffsetComponent*> refs(components_.size());
    for (size_t i = 0; i < components_.size(); ++i) refs[i] = &components_[i];

    buildObbTreeRecursive_(refs, 0, (int)refs.size(), 0, obbTree_);

    componentRefs_ = std::move(refs);

    timeTree = std::chrono::duration<double>(Clock::now() - t0).count();
    logStep("OBB_TREE", "Done  Depth estimate: " +
            std::to_string((int)std::ceil(std::log2(components_.size()))) +
            "  Time: " + std::to_string(timeTree) + " s");
}

// ============================================================
// Ray-component intersection tests
// ============================================================
static bool raySphereIntersect(const Vec3& ro, const Vec3& rd,
                               const Vec3& center, double radius,
                               double& tOut, Vec3& normalOut)
{
    Vec3 oc = ro - center;
    double b = oc.dot(rd);
    double c = oc.dot(oc) - radius * radius;
    double disc = b * b - c;
    if (disc < 0) return false;

    double s = std::sqrt(disc);
    double t = -b - s;          // nearest intersection
    if (t < 1e-9) t = -b + s;  // try far intersection if inside
    if (t < 1e-9) return false;

    tOut    = t;
    normalOut = ((ro + rd * t) - center).normalized();
    return true;
}

static bool rayCylinderIntersect(const Vec3& ro, const Vec3& rd,
                                 const Vec3& A, const Vec3& B, double R,
                                 double& tOut, Vec3& normalOut)
{
    Vec3  AB  = B - A;
    Vec3  AO  = ro - A;
    Vec3  dAB = AB;
    double lenAB = AB.length();
    if (lenAB < 1e-12) return false;
    dAB = dAB / lenAB;

    Vec3  rdC  = rd - dAB * rd.dot(dAB);
    Vec3  AOC  = AO - dAB * AO.dot(dAB);
    double a   = rdC.dot(rdC);
    double b_  = 2.0 * rdC.dot(AOC);
    double c_  = AOC.dot(AOC) - R * R;
    double disc = b_ * b_ - 4.0 * a * c_;
    if (disc < 1e-12) return false;

    double sq = std::sqrt(disc);
    double t1 = (-b_ - sq) / (2.0 * a);
    double t2 = (-b_ + sq) / (2.0 * a);

    for (double t : {t1, t2}) {
        if (t < 1e-9) continue;
        Vec3 P  = ro + rd * t;
        double s = (P - A).dot(dAB);
        if (s < -R || s > lenAB + R) continue;  // beyond segment (+/- R for cap)

        tOut      = t;
        Vec3 axisPt = A + dAB * s;
        normalOut = (P - axisPt).normalized();
        return true;
    }
    return false;
}

static bool raySlabIntersect(const Vec3& ro, const Vec3& rd,
                             const Vec3& origin, const Vec3& nrm,
                             const Vec3& u, const Vec3& v, double thickness,
                             double& tOut, Vec3& normalOut)
{
    double denom = nrm.dot(rd);
    if (std::abs(denom) < 1e-12) return false;

    double halfT = thickness * 0.5;
    double d0 = (origin - ro).dot(nrm) + halfT;
    double d1 = (origin - ro).dot(nrm) - halfT;

    // Two slab faces
    double tNear = (d0 > 0) ? d0 / denom : -1e18;
    Vec3   nNear = nrm;
    double tFar  = (d1 > 0) ? d1 / denom : -1e18;
    Vec3   nFar  = -nrm;

    if (tNear > tFar) { std::swap(tNear, tFar); std::swap(nNear, nFar); }
    if (tFar < 1e-9) return false;

    double t = (tNear > 1e-9) ? tNear : tFar;
    if (t < 1e-9) return false;

    // Verify inside triangle bounds (project onto u,v)
    Vec3 P = ro + rd * t - origin;
    double pu = P.dot(u) / u.dot(u);
    double pv = P.dot(v) / v.dot(v);

    if (pu >= -0.1 && pv >= -0.1 && pu + pv <= 1.1) {
        tOut      = t;
        normalOut = (t == tNear) ? nNear : nFar;
        return true;
    }
    return false;
}

// ============================================================
// CollectIntersections -- traverse OBB tree, test each component
// ============================================================
void SdfOffset::collectIntersections_(const OBBNode* node, const Vec3& ro, const Vec3& rd,
                                      std::vector<HitRecord>& hits) const
{
    if (!node) return;

    double tmin, tmax;
    if (!node->bounds.intersect(ro, rd, tmin, tmax)) return;

    if (node->isLeaf) {
        for (int i = 0; i < node->count; ++i) {
            const OffsetComponent& c = *componentRefs_[node->firstIdx + i];
            double t = -1;
            Vec3 n;

            bool hit = false;
            switch (c.type) {
                case OffsetComponentType::Sphere:
                    hit = raySphereIntersect(ro, rd, c.center, c.radius, t, n);
                    break;
                case OffsetComponentType::Cylinder:
                    hit = rayCylinderIntersect(ro, rd, c.cylA, c.cylB, c.cylRadius, t, n);
                    break;
                case OffsetComponentType::Slab:
                    hit = raySlabIntersect(ro, rd, c.slabOrigin, c.slabNormal,
                                           c.slabU, c.slabV, c.slabThickness, t, n);
                    break;
            }
            if (hit && t > 1e-9) {
                hits.push_back({t, ro + rd * t, n});
            }
        }
        return;
    }

    collectIntersections_(node->left.get(),  ro, rd, hits);
    collectIntersections_(node->right.get(), ro, rd, hits);
}

// ============================================================
// ComputeSdf -- main pipeline (paper Algorithm 1 + applications)
// ============================================================
void SdfOffset::computeSdf()
{
    auto t0 = Clock::now();
    logStep("SDF", "Computing offset-surface SDF for " + std::to_string(vertices.size()) + " vertices...");

    sdfValues.assign(vertices.size(), 0.0);
    int missedCount = 0;

    for (size_t vi = 0; vi < vertices.size(); ++vi) {
        Vec3  p   = vertices[vi];
        Vec3  n   = vertexNormals[vi];
        Vec3  rd  = n;   // ray direction = outward normal

        // Collect all intersections along this ray
        std::vector<HitRecord> hits;
        hits.reserve(32);
        collectIntersections_(obbTree_.get(), p, rd, hits);

        if (hits.size() >= 2) {
            std::sort(hits.begin(), hits.end(),
                      [](const HitRecord& a, const HitRecord& b) { return a.t < b.t; });

            // q  = hits[0].point   (outer offset surface)
            // q' = hits.back().point (inner offset surface)
            // SDF = ||q' - q|| - 2r
            double D = (hits.back().point - hits[0].point).length();
            sdfValues[vi] = D - 2.0 * offsetRadius;
        } else if (hits.size() == 1) {
            // Only one intersection -- use single-side estimate
            sdfValues[vi] = hits[0].t - offsetRadius;
        } else {
            // No intersections -- fallback: closest component distance
            double minDist = 1e18;
            for (const auto& comp : components_) {
                double d = 0;
                switch (comp.type) {
                    case OffsetComponentType::Sphere:
                        d = (p - comp.center).length() - comp.radius;
                        break;
                    case OffsetComponentType::Cylinder: {
                        Vec3 AB = comp.cylB - comp.cylA;
                        double t = (p - comp.cylA).dot(AB) / AB.dot(AB);
                        t = std::clamp(t, 0.0, 1.0);
                        Vec3 closest = comp.cylA + AB * t;
                        d = (p - closest).length() - comp.cylRadius;
                        break;
                    }
                    default: break;
                }
                minDist = std::min(minDist, d);
            }
            sdfValues[vi] = minDist;
            missedCount++;
        }
    }

    // Normalize to [0, 1] using bounding-box diagonal (paper Section 5)
    if (!sdfValues.empty()) {
        double sdfMin = *std::min_element(sdfValues.begin(), sdfValues.end());
        double sdfMax = *std::max_element(sdfValues.begin(), sdfValues.end());
        double range  = sdfMax - sdfMin;

        logStep("SDF", "Raw SDF range: [" + std::to_string(sdfMin) + ", " + std::to_string(sdfMax) + "]");

        for (auto& v : sdfValues)
            v = (range > 1e-12) ? (v - sdfMin) / range : 0.0;
    }

    timeSdf = std::chrono::duration<double>(Clock::now() - t0).count();
    logStep("SDF", "Missed rays: " + std::to_string(missedCount) + "/" + std::to_string(vertices.size()));
    logStep("SDF", "Done  Time: " + std::to_string(timeSdf) + " s");
}
