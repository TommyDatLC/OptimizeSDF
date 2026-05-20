#define PI 3.14159265359f
#define GOLDEN_RATIO 1.61803398875f

typedef struct {
    uint numTriangles;
    uint triangleStartIndex;
} LeafData;

// Cấu trúc Nút Octree trên GPU chứa Bounding Box
typedef struct {
    float4 bboxMin; // xyz = Min Box, w = Descriptor (Pointer + Mask)
    float4 bboxMax; // xyz = Max Box, w = unused
} GPUNode;

// Thuật toán cắt tia - tam giác
bool RayTriangleIntersect(float3 orig, float3 dir, float3 v0, float3 v1, float3 v2, float* t, float3* hitNormal) {
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(dir, edge2);
    float a = dot(edge1, h);
    if (a > -1e-5f && a < 1e-5f) return false;

    float f = 1.0f / a;
    float3 s = orig - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    float3 q = cross(s, edge1);
    float v = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    float current_t = f * dot(edge2, q);
    if (current_t > 1e-5f) {
        *t = current_t;
        *hitNormal = normalize(cross(edge1, edge2));
        return true;
    }
    return false;
}

// ==============================================================
// PASS 1: BẮN TIA VÀ DUYỆT CÂY (RAYCASTING)
// ĐÃ SỬA: Xóa hoàn toàn debugData, bây giờ hàm CHỈ NHẬN CHÍNH XÁC 11 THAM SỐ
// ==============================================================
__kernel void sdf_raycast(
    __global const float4* triangleTexture, // 0
    __global const GPUNode* octreeNodes,    // 1
    __global const LeafData* leafDataArray, // 2
    __global const float4* origins,         // 3
    __global const float4* normals,         // 4
    __global const float4* tangents,        // 5
    __global const float4* binormals,       // 6
    __global float* outDistances,           // 7
    const uint numRaysPerPoint,             // 8
    const float coneAmplitude,              // 9
    const float maxSize                     // 10
) {
    int global_id = get_global_id(0);
    int point_idx = global_id / numRaysPerPoint;
    int ray_idx = global_id % numRaysPerPoint;

    float min_closeness = (maxSize / 2.0f) * 0.00001f;

    float3 p = (float3)(origins[point_idx].x, origins[point_idx].y, origins[point_idx].z);
    float3 n = (float3)(normals[point_idx].x, normals[point_idx].y, normals[point_idx].z);
    float3 t_vec = (float3)(tangents[point_idx].x, tangents[point_idx].y, tangents[point_idx].z);
    float3 b = (float3)(binormals[point_idx].x, binormals[point_idx].y, binormals[point_idx].z);

    // Xoay tia ngẫu nhiên bằng Spherical Fibonacci
    float z_min = cos(coneAmplitude / 2.0f);
    float z = 1.0f - (1.0f - z_min) * ((float)ray_idx / (float)(numRaysPerPoint - 1));
    float phi = 2.0f * PI * ((float)ray_idx / GOLDEN_RATIO);

    float radius = sqrt(1.0f - z * z);
    float3 localDir = (float3)(radius * cos(phi), radius * sin(phi), z);

    float3 rayDir = normalize(localDir.x * t_vec + localDir.y * b + localDir.z * n);

    // Tính nghịch đảo hướng tia (Chống chia cho 0)
    float3 rayInvDir = (float3)(
        (fabs(rayDir.x) > 1e-8f) ? 1.0f / rayDir.x : 1e8f,
        (fabs(rayDir.y) > 1e-8f) ? 1.0f / rayDir.y : 1e8f,
        (fabs(rayDir.z) > 1e-8f) ? 1.0f / rayDir.z : 1e8f
    );

    uint stack[32];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    float closest_t = -1.0f;
    bool hit_found = false;

    while (stackPtr > 0) {
        uint nodeIndex = stack[--stackPtr];
        GPUNode node = octreeNodes[nodeIndex];

        // -----------------------------------------------------------
        // RAY-AABB INTERSECTION TEST
        // -----------------------------------------------------------
        float3 boxMin = (float3)(node.bboxMin.x, node.bboxMin.y, node.bboxMin.z);
        float3 boxMax = (float3)(node.bboxMax.x, node.bboxMax.y, node.bboxMax.z);

        float3 t0 = (boxMin - p) * rayInvDir;
        float3 t1 = (boxMax - p) * rayInvDir;
        float3 tmin3 = fmin(t0, t1);
        float3 tmax3 = fmax(t0, t1);

        float tnear = fmax(fmax(tmin3.x, tmin3.y), tmin3.z);
        float tfar  = fmin(fmin(tmax3.x, tmax3.y), tmax3.z);

        float max_t_allowed = (closest_t > 0.0f) ? closest_t : 100000.0f;

        if (tnear > tfar || tfar < 0.0f || tnear > max_t_allowed) {
            continue; // Bỏ qua toàn bộ Node này vì tia không bay xuyên qua hộp
        }

        uint descriptor = as_uint(node.bboxMin.w);
        uint childPointer = descriptor >> 8;
        uchar validMask = descriptor & 0xFF;

        if (validMask == 0) {
            // NÚT LÁ
            LeafData leaf = leafDataArray[childPointer];
            for (uint i = 0; i < leaf.numTriangles; ++i) {
                uint triIdx = leaf.triangleStartIndex + i;

                float4 v0_4 = triangleTexture[triIdx * 3 + 0];
                float4 v1_4 = triangleTexture[triIdx * 3 + 1];
                float4 v2_4 = triangleTexture[triIdx * 3 + 2];

                float3 v0 = (float3)(v0_4.x, v0_4.y, v0_4.z);
                float3 v1 = (float3)(v1_4.x, v1_4.y, v1_4.z);
                float3 v2 = (float3)(v2_4.x, v2_4.y, v2_4.z);

                float t;
                float3 hitNormal;
                if (RayTriangleIntersect(p, rayDir, v0, v1, v2, &t, &hitNormal)) {
                    if (t > min_closeness) {
                        if (!hit_found || t < closest_t) {
                            closest_t = t;
                            hit_found = true;
                        }
                    }
                }
            }
        } else {
            // NÚT NHÁNH
            for (int i = 0; i < 8; ++i) {
                if ((validMask & (1 << i)) != 0) {
                    stack[stackPtr++] = childPointer + i;
                }
            }
        }
    }

    outDistances[global_id] = closest_t;
}

// ==============================================================
// PASS 2: TÍNH TRUNG BÌNH SDF TRÊN GPU
// ==============================================================
__kernel void sdf_average(
    __global const float* outDistances,
    __global float* finalSDF,
    const uint raysPerPoint,
    const uint numVertices
) {
    int vertex_idx = get_global_id(0);
    if (vertex_idx >= numVertices) return;

    float sum = 0.0f;
    int validRays = 0;
    int start_idx = vertex_idx * raysPerPoint;

    for (int r = 0; r < raysPerPoint; ++r) {
        float dist = outDistances[start_idx + r];
        if (dist > 0.0f) {
            sum += dist;
            validRays++;
        }
    }
    finalSDF[vertex_idx] = (validRays > 0) ? (sum / validRays) : 0.0f;
}