#ifndef OCTREE_BUILDER_HPP
#define OCTREE_BUILDER_HPP

#include <vector>
#include <memory>
#include <cmath>
#include <queue>
#include <iostream>
#include <cstring>
#include "../../Core/Model.cuh" // Chỉnh lại đường dẫn nếu cần

// Định nghĩa cơ bản để giao tiếp với OpenCL
struct HostFloat4 { float x, y, z, w; };
struct LeafData { uint32_t numTriangles; uint32_t triangleStartIndex; };

namespace Octree {

    struct Vec3 { float x, y, z; };
    struct AABB { Vec3 min, max; };
    struct Triangle { Vec3 v0, v1, v2; };

    // Cấu trúc Node xuất ra GPU (32 bytes)
    struct GPUNode {
        HostFloat4 bboxMin = {0.0f, 0.0f, 0.0f, 0.0f}; // x,y,z = Min Box. w = Descriptor
        HostFloat4 bboxMax = {0.0f, 0.0f, 0.0f, 0.0f}; // x,y,z = Max Box.
    };

    // Các hàm Toán học Vector cơ bản
    inline Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    inline Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; }
    inline float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

    inline void findMinMax(float x0, float x1, float x2, float& min, float& max) {
        min = max = x0;
        if (x1 < min) min = x1; if (x1 > max) max = x1;
        if (x2 < min) min = x2; if (x2 > max) max = x2;
    }

    // =========================================================================
    // THUẬT TOÁN MÖLLER AABB - TRIANGLE INTERSECTION
    // =========================================================================
    inline bool planeBoxOverlap(Vec3 normal, Vec3 vert, Vec3 maxbox) {
        Vec3 vmin, vmax;
        if (normal.x > 0.0f) { vmin.x = -maxbox.x - vert.x; vmax.x = maxbox.x - vert.x; }
        else { vmin.x = maxbox.x - vert.x; vmax.x = -maxbox.x - vert.x; }
        if (normal.y > 0.0f) { vmin.y = -maxbox.y - vert.y; vmax.y = maxbox.y - vert.y; }
        else { vmin.y = maxbox.y - vert.y; vmax.y = -maxbox.y - vert.y; }
        if (normal.z > 0.0f) { vmin.z = -maxbox.z - vert.z; vmax.z = maxbox.z - vert.z; }
        else { vmin.z = maxbox.z - vert.z; vmax.z = -maxbox.z - vert.z; }
        if (dot(normal, vmin) > 0.0f) return false;
        if (dot(normal, vmax) >= 0.0f) return true;
        return false;
    }

    #define AXISTEST_X01(a, b, fa, fb) \
        p0 = a*v0.y - b*v0.z; p2 = a*v2.y - b*v2.z; \
        if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
        rad = fa * boxhalfsize.y + fb * boxhalfsize.z; \
        if(min>rad || max<-rad) return false;

    #define AXISTEST_Y02(a, b, fa, fb) \
        p0 = -a*v0.x + b*v0.z; p2 = -a*v2.x + b*v2.z; \
        if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
        rad = fa * boxhalfsize.x + fb * boxhalfsize.z; \
        if(min>rad || max<-rad) return false;

    #define AXISTEST_Z12(a, b, fa, fb) \
        p1 = a*v1.x - b*v1.y; p2 = a*v2.x - b*v2.y; \
        if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;} \
        rad = fa * boxhalfsize.x + fb * boxhalfsize.y; \
        if(min>rad || max<-rad) return false;

    inline bool TriBoxOverlap(Vec3 boxcenter, Vec3 boxhalfsize, Triangle t) {
        Vec3 v0 = sub(t.v0, boxcenter);
        Vec3 v1 = sub(t.v1, boxcenter);
        Vec3 v2 = sub(t.v2, boxcenter);
        Vec3 e0 = sub(v1, v0), e1 = sub(v2, v1), e2 = sub(v0, v2);
        float min, max, p0, p1, p2, rad, fex, fey, fez;

        fex = std::abs(e0.x); fey = std::abs(e0.y); fez = std::abs(e0.z);
        AXISTEST_X01(e0.z, e0.y, fez, fey);
        AXISTEST_Y02(e0.z, e0.x, fez, fex);
        AXISTEST_Z12(e0.y, e0.x, fey, fex);

        fex = std::abs(e1.x); fey = std::abs(e1.y); fez = std::abs(e1.z);
        AXISTEST_X01(e1.z, e1.y, fez, fey);
        AXISTEST_Y02(e1.z, e1.x, fez, fex);
        AXISTEST_Z12(e1.y, e1.x, fey, fex);

        fex = std::abs(e2.x); fey = std::abs(e2.y); fez = std::abs(e2.z);
        AXISTEST_X01(e2.z, e2.y, fez, fey);
        AXISTEST_Y02(e2.z, e2.x, fez, fex);
        AXISTEST_Z12(e2.y, e2.x, fey, fex);

        findMinMax(v0.x, v1.x, v2.x, min, max); if (min > boxhalfsize.x || max < -boxhalfsize.x) return false;
        findMinMax(v0.y, v1.y, v2.y, min, max); if (min > boxhalfsize.y || max < -boxhalfsize.y) return false;
        findMinMax(v0.z, v1.z, v2.z, min, max); if (min > boxhalfsize.z || max < -boxhalfsize.z) return false;

        Vec3 normal = cross(e0, e1);
        if (!planeBoxOverlap(normal, v0, boxhalfsize)) return false;
        return true;
    }

    struct OctreeNode {
        AABB box;
        std::vector<int> triangleIndices;
        std::unique_ptr<OctreeNode> children[8];
        bool isLeaf = true;
    };

    class Builder {
    private:
        const std::vector<Triangle>& allTriangles;
        int maxDepth;
        int maxTrianglesPerLeaf;

        void Subdivide(OctreeNode* node, int currentDepth) {
            if (currentDepth >= maxDepth || node->triangleIndices.size() <= maxTrianglesPerLeaf) {
                return;
            }

            node->isLeaf = false;
            Vec3 center = { (node->box.min.x + node->box.max.x) * 0.5f,
                            (node->box.min.y + node->box.max.y) * 0.5f,
                            (node->box.min.z + node->box.max.z) * 0.5f };
            Vec3 halfSize = { (node->box.max.x - node->box.min.x) * 0.25f,
                              (node->box.max.y - node->box.min.y) * 0.25f,
                              (node->box.max.z - node->box.min.z) * 0.25f };

            for (int i = 0; i < 8; ++i) {
                node->children[i] = std::make_unique<OctreeNode>();
                Vec3 childMin, childMax;
                childMin.x = (i & 1) ? center.x : node->box.min.x;
                childMax.x = (i & 1) ? node->box.max.x : center.x;
                childMin.y = (i & 2) ? center.y : node->box.min.y;
                childMax.y = (i & 2) ? node->box.max.y : center.y;
                childMin.z = (i & 4) ? center.z : node->box.min.z;
                childMax.z = (i & 4) ? node->box.max.z : center.z;

                node->children[i]->box = { childMin, childMax };
                Vec3 childCenter = { (childMin.x + childMax.x) * 0.5f,
                                     (childMin.y + childMax.y) * 0.5f,
                                     (childMin.z + childMax.z) * 0.5f };

                for (int triIdx : node->triangleIndices) {
                    if (TriBoxOverlap(childCenter, halfSize, allTriangles[triIdx])) {
                        node->children[i]->triangleIndices.push_back(triIdx);
                    }
                }

                if (node->children[i]->triangleIndices.empty()) {
                    node->children[i].reset();
                } else {
                    Subdivide(node->children[i].get(), currentDepth + 1);
                }
            }

            node->triangleIndices.clear();
            node->triangleIndices.shrink_to_fit();
        }

    public:
        Builder(const std::vector<Triangle>& triangles, int depth, int maxTris)
            : allTriangles(triangles), maxDepth(depth), maxTrianglesPerLeaf(maxTris) {}

        std::unique_ptr<OctreeNode> BuildTree(AABB rootBox) {
            auto root = std::make_unique<OctreeNode>();
            root->box = rootBox;
            for (int i = 0; i < allTriangles.size(); ++i) {
                root->triangleIndices.push_back(i);
            }
            Subdivide(root.get(), 0);
            return root;
        }
    };

    inline void FlattenOctree(
        const OctreeNode* root,
        const std::vector<Triangle>& allTriangles,
        std::vector<GPUNode>& outOctreeNodes,
        std::vector<LeafData>& outLeafData,
        std::vector<HostFloat4>& outTriangleTexture)
    {
        outOctreeNodes.clear();
        outLeafData.clear();
        outTriangleTexture.clear();

        struct QueueItem { const OctreeNode* node; int flatIndex; };
        std::queue<QueueItem> q;

        outOctreeNodes.push_back(GPUNode());
        q.push({root, 0});

        while (!q.empty()) {
            auto item = q.front(); q.pop();
            const OctreeNode* curr = item.node;

            uint32_t descriptor = 0;

            if (curr->isLeaf) {
                LeafData leaf;
                leaf.triangleStartIndex = outTriangleTexture.size() / 3;
                leaf.numTriangles = curr->triangleIndices.size();

                for (int triIdx : curr->triangleIndices) {
                    Triangle t = allTriangles[triIdx];
                    outTriangleTexture.push_back({t.v0.x, t.v0.y, t.v0.z, 0.0f});
                    outTriangleTexture.push_back({t.v1.x, t.v1.y, t.v1.z, 0.0f});
                    outTriangleTexture.push_back({t.v2.x, t.v2.y, t.v2.z, 0.0f});
                }
                outLeafData.push_back(leaf);

                descriptor = ((outLeafData.size() - 1) << 8) | 0;
            } else {
                int childPointer = outOctreeNodes.size();
                outOctreeNodes.resize(outOctreeNodes.size() + 8, GPUNode());

                uint8_t validMask = 0;
                for (int i = 0; i < 8; ++i) {
                    if (curr->children[i] != nullptr) {
                        validMask |= (1 << i);
                        q.push({curr->children[i].get(), childPointer + i});
                    }
                }
                descriptor = (childPointer << 8) | validMask;
            }

            float descF;
            std::memcpy(&descF, &descriptor, sizeof(uint32_t));

            outOctreeNodes[item.flatIndex].bboxMin = {curr->box.min.x, curr->box.min.y, curr->box.min.z, descF};
            outOctreeNodes[item.flatIndex].bboxMax = {curr->box.max.x, curr->box.max.y, curr->box.max.z, 0.0f};
        }
    }

} // end namespace

#endif // OCTREE_BUILDER_HPP