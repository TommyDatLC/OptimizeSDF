#pragma once

#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <limits>
#include <memory>
#include <algorithm>

// ============================================================
// Vec3 -- basic 3D vector (reuses pattern from MathHelper.cu)
// ============================================================
struct Vec3 {
    double x = 0, y = 0, z = 0;

    Vec3() = default;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    explicit Vec3(const std::array<double, 3>& a) : x(a[0]), y(a[1]), z(a[2]) {}

    Vec3 operator+(const Vec3& b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec3 operator-(const Vec3& b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec3 operator-()              const { return {-x, -y, -z}; }
    Vec3 operator*(double s)      const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s)      const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }

    double length()   const { return std::sqrt(x * x + y * y + z * z); }
    double lengthSq() const { return x * x + y * y + z * z; }

    Vec3 normalized() const {
        double len = length();
        if (len > 1e-12) return *this / len;
        return {0, 0, 1};
    }

    double dot(const Vec3& b)   const { return x * b.x + y * b.y + z * b.z; }
    Vec3   cross(const Vec3& b) const {
        return {y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x};
    }
};

// ============================================================
// Triangle (reuses pattern from Model.cu ReadFromObjFile)
// ============================================================
struct Triangle {
    size_t i0, i1, i2;     // indices into the vertex array
};

// ============================================================
// AABB -- axis-aligned bounding box
// ============================================================
struct AABB {
    Vec3 bmin{ std::numeric_limits<double>::max(),
               std::numeric_limits<double>::max(),
               std::numeric_limits<double>::max() };
    Vec3 bmax{ -std::numeric_limits<double>::max(),
               -std::numeric_limits<double>::max(),
               -std::numeric_limits<double>::max() };

    Vec3 center()  const { return (bmin + bmax) * 0.5; }
    Vec3 extent()  const { return bmax - bmin; }
    Vec3 diagonal()const { return extent(); }

    void expand(const Vec3& p) {
        bmin.x = std::min(bmin.x, p.x); bmax.x = std::max(bmax.x, p.x);
        bmin.y = std::min(bmin.y, p.y); bmax.y = std::max(bmax.y, p.y);
        bmin.z = std::min(bmin.z, p.z); bmax.z = std::max(bmax.z, p.z);
    }
    void expandPad(const Vec3& p, double r) {
        bmin.x = std::min(bmin.x, p.x - r); bmax.x = std::max(bmax.x, p.x + r);
        bmin.y = std::min(bmin.y, p.y - r); bmax.y = std::max(bmax.y, p.y + r);
        bmin.z = std::min(bmin.z, p.z - r); bmax.z = std::max(bmax.z, p.z + r);
    }
    void merge(const AABB& o) { expand(o.bmin); expand(o.bmax); }

    double surfaceArea() const {
        Vec3 e = extent();
        return 2.0 * (e.x * e.y + e.y * e.z + e.z * e.x);
    }

    // Ray-AABB intersection (slab method, same pattern as OptiX BVH traversal)
    bool intersect(const Vec3& ro, const Vec3& rd, double& tmin, double& tmax) const;
};

// ============================================================
// Offset surface components (paper Section 4)
//   For each triangle: 3 spheres + 3 cylinders + 1 slab
// ============================================================
enum class OffsetComponentType { Sphere, Cylinder, Slab };

struct OffsetComponent {
    OffsetComponentType type;
    AABB obb;

    // Sphere
    Vec3 center;
    double radius = 0;

    // Cylinder
    Vec3 cylA, cylB;          // endpoints of the center-line segment
    Vec3 cylDir;               // unit direction B - A
    double cylLen = 0;         // length of segment
    double cylRadius = 0;

    // Slab (tri-prism approximation)
    Vec3 slabNormal;           // face normal
    Vec3 slabU, slabV;         // local tangent frame
    Vec3 slabOrigin;           // v0 of the triangle
    double slabThickness = 0;  // 2 * offsetRadius (total thickness)
};

// ============================================================
// OBB Tree node -- BVH over offset components
// ============================================================
struct OBBNode {
    AABB bounds;
    bool isLeaf = false;
    int firstIdx = 0, count = 0;          // leaf: range in flat array
    std::unique_ptr<OBBNode> left, right;  // internal
};

// ============================================================
// Ray hit record
// ============================================================
struct HitRecord {
    double t       = -1;
    Vec3   point;
    Vec3   normal;   // outward normal of the offset surface component
};

// ============================================================
// SdfOffset -- the main class implementing the paper's algorithm
// ============================================================
class SdfOffset {
public:
    // ----- data (reuses pattern from Model class) -----
    std::vector<Vec3>               vertices;
    std::vector<Triangle>           triangles;
    std::vector<Vec3>               vertexNormals;
    std::vector<double>             sdfValues;

    double offsetRadius = 0.05;     // fraction of bounding-box diagonal (default 5%)

    // ----- pipeline steps with timing -----
    void loadObj(const std::string& filename);
    void computeVertexNormals();     // reuses pattern from ModelHelper.cu
    void buildOffsetComponents();
    void buildObbTree();
    void computeSdf();               // main SDF algorithm

    // ----- timing logs (seconds) -----
    double timeLoad = 0, timeNormals = 0, timeOffset = 0;
    double timeTree = 0, timeSdf = 0, timeTotal = 0;

private:
    std::vector<OffsetComponent>   components_;
    std::vector<OffsetComponent*>  componentRefs_;   // tree-order pointers
    std::unique_ptr<OBBNode>       obbTree_;

    void computeComponentObb_(OffsetComponent& c);
    void buildObbTreeRecursive_(std::vector<OffsetComponent*>& refs,
                                int begin, int end, int depth,
                                std::unique_ptr<OBBNode>& node);
    void collectIntersections_(const OBBNode* node, const Vec3& ro, const Vec3& rd,
                               std::vector<HitRecord>& hits) const;
};
