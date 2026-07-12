#ifndef SDF_KERNELS_CUH
#define SDF_KERNELS_CUH

#include <cuda_runtime.h>
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#include <cub/cub.cuh>
#include <cmath>

// =========================================================================================
// ATOMIC MIN/MAX CHO FLOAT
// Do phần cứng GPU (trừ các thế hệ siêu mới) không hỗ trợ hàm atomicMin/Max trực tiếp 
// cho kiểu số thực (float), ta phải tự mô phỏng nó bằng vòng lặp Compare-And-Swap (CAS).
// =========================================================================================

inline __device__ __forceinline__ void atomicMinFloat(float* addr, float value) {
    // Ép kiểu con trỏ số thực (float*) thành con trỏ số nguyên (int*) 
    // vì hàm phần cứng atomicCAS chỉ chấp nhận kiểu số nguyên 32-bit.
    int* address_as_int = (int*)addr;
    
    // Đọc giá trị hiện tại từ VRAM (dưới dạng bit số nguyên) và lưu vào biến old.
    int old = *address_as_int, assumed;
    
    // Vòng lặp thử và ghi lại (Optimistic Concurrency Control)
    do {
        // Lưu lại giá trị giả định (assumed) trước khi kiểm tra
        assumed = old;
        
        // Chuyển bit-pattern số nguyên sang float để so sánh. 
        // Nếu giá trị trong VRAM đã nhỏ hơn hoặc bằng giá trị mới, ta không cần cập nhật -> Thoát.
        if (__int_as_float(assumed) <= value) break;
        
        // Cố gắng ghi đè giá trị mới (__float_as_int(value)).
        // Nếu tại thời điểm ghi, giá trị trong VRAM vẫn bằng 'assumed' (không bị thread khác tranh ghi), 
        // atomicCAS sẽ ghi thành công và trả về 'assumed'. Lúc này assumed == old -> Thoát vòng lặp.
        // Nếu bị thread khác ghi đè trước, atomicCAS trả về giá trị thực tế mới của VRAM (khác assumed).
        // Vòng lặp sẽ tiếp tục để tính toán lại với giá trị mới này.
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

inline __device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
    // Ép kiểu con trỏ float* thành int* để sử dụng hàm atomicCAS
    int* address_as_int = (int*)addr;
    
    // Đọc giá trị bit hiện tại tại địa chỉ bộ nhớ
    int old = *address_as_int, assumed;
    
    // Bắt đầu vòng lặp Compare-And-Swap
    do {
        // Lưu lại giá trị hiện tại làm mốc giả định
        assumed = old;
        
        // So sánh: Nếu giá trị thực tế trong bộ nhớ đã LỚN HƠN hoặc BẰNG giá trị cần cập nhật,
        // thì không cần làm gì thêm, thoát khỏi vòng lặp ngay lập tức.
        if (__int_as_float(assumed) >= value) break;
        
        // Thực hiện ghi nguyên tử (atomic write):
        // Chỉ ghi '__float_as_int(value)' vào bộ nhớ nếu giá trị tại đó vẫn đang bằng 'assumed'.
        // Trả về giá trị thực tế của bộ nhớ sau lệnh CAS.
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
        
    // Lặp lại nếu ghi thất bại (assumed != old tức là có thread khác đã chèn ngang thay đổi bộ nhớ)
    } while (assumed != old);
}

// =========================================================================================
// KERNEL: BOUNDING BOX & NORMALIZATION
// =========================================================================================
inline __global__ void GPUComputeBoundingBox(const float3* vertices, int numVertices, float* d_minMax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;
    float3 v = vertices[idx];
    atomicMinFloat(&d_minMax[0], v.x); atomicMaxFloat(&d_minMax[1], v.x);
    atomicMinFloat(&d_minMax[2], v.y); atomicMaxFloat(&d_minMax[3], v.y);
    atomicMinFloat(&d_minMax[4], v.z); atomicMaxFloat(&d_minMax[5], v.z);
}

inline __global__ void GPUComputeSDFMinMax(const float* rawSDF, int numVertices, float* d_minMaxSDF) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;
    float val = rawSDF[idx];
    atomicMinFloat(&d_minMaxSDF[0], val);
    atomicMaxFloat(&d_minMaxSDF[1], val);
}

inline __global__ void GPUApplySDFNormalization(float* rawSDF, int numVertices, const float* d_minMaxSDF) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;
    
    float minSDF = d_minMaxSDF[0];
    float maxSDF = d_minMaxSDF[1];
    float alpha = 4.0f;
    float logAlphaPlus1 = logf(alpha + 1.0f);
    
    if (maxSDF - minSDF > 1e-6f) {
        float normalized = (rawSDF[idx] - minSDF) / (maxSDF - minSDF);
        rawSDF[idx] = logf(normalized * alpha + 1.0f) / logAlphaPlus1;
    } else {
        rawSDF[idx] = 0.0f;
    }
}

// =========================================================================================
// CÁC KERNEL HỖ TRỢ TÍNH TOÁN SDF BẰNG CUB
// =========================================================================================
inline __global__ void GPUGenerateOffsets(int* d_offsets, int numVertices, int raysPerPoint) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= numVertices) {
        d_offsets[idx] = idx * raysPerPoint;
    }
}

inline __global__ void GPUComputeRawSDF(
    float* d_distances,
    float* d_weights,
    const int* d_validCounts,
    float* d_rawSDF,
    int numVertices,
    int raysPerPoint) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    int validCount = d_validCounts[idx];
    if (validCount == 0) {
        d_rawSDF[idx] = 0.0f;
        return;
    }

    int baseIdx = idx * raysPerPoint;

    float totalWeightedDist = 0.0f;
    float totalWeight = 0.0f;

    // Không dùng Outlier Removal: Tính thẳng trung bình có trọng số của TẤT CẢ các tia
    for (int i = 0; i < validCount; i++) {
        float dist = d_distances[baseIdx + i];
        float weight = d_weights[baseIdx + i];
        
        totalWeightedDist += dist * weight;
        totalWeight += weight;
    }
    
    d_rawSDF[idx] = (totalWeight > 0.0f) ? (totalWeightedDist / totalWeight) : 0.0f;
}

inline __global__ void GPUGenerateEdges(const uint3* indices, int numFaces, uint64_t* d_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numFaces) return;
    uint3 face = indices[idx];
    int base = idx * 6;
    d_edges[base + 0] = ((uint64_t)face.x << 32) | face.y;
    d_edges[base + 1] = ((uint64_t)face.x << 32) | face.z;
    d_edges[base + 2] = ((uint64_t)face.y << 32) | face.x;
    d_edges[base + 3] = ((uint64_t)face.y << 32) | face.z;
    d_edges[base + 4] = ((uint64_t)face.z << 32) | face.x;
    d_edges[base + 5] = ((uint64_t)face.z << 32) | face.y;
}

/**
 * KERNEL: GPUExtractCSR
 * 
 * THAM SỐ:
 * - d_uniqueEdges: [Input] Mảng chứa các cạnh có hướng duy nhất đã được sắp xếp tăng dần theo đỉnh nguồn (src).
 *                  Mỗi cạnh được nén trong 1 số uint64_t (32 bit cao là src, 32 bit thấp là dst).
 * - numUniqueEdges: [Input] Tổng số lượng cạnh duy nhất sau khi đã lọc trùng.
 * - d_nbrOffsets: [Output] Mảng Offset của đồ thị CSR (kích thước = numVertices + 1).
 *                 d_nbrOffsets[i] sẽ lưu vị trí bắt đầu các đỉnh kề của đỉnh i trong mảng d_nbrLists.
 * - d_nbrLists: [Output] Mảng danh sách kề thực tế (kích thước = numUniqueEdges), chứa ID của các đỉnh kề (dst).
 * - numVertices: [Input] Tổng số đỉnh của lưới mesh.
 * 
 * TẠI SAO LÀM NHƯ VẬY:
 * Kernel này phân tích mảng cạnh đã sắp xếp để dựng cấu trúc đồ thị CSR song song. Bằng cách so sánh
 * đỉnh nguồn của cạnh hiện tại (idx) với cạnh trước đó (idx - 1), kernel phát hiện ra ranh giới (biên)
 * nơi danh sách đỉnh kề của đỉnh cũ kết thúc và đỉnh mới bắt đầu, từ đó ghi nhận offset một cách chính xác
 * mà hoàn toàn không xảy ra tranh chấp bộ nhớ (race condition).
 */
inline __global__ void GPUExtractCSR(const uint64_t* d_uniqueEdges, int numUniqueEdges, int* d_nbrOffsets, int* d_nbrLists, int numVertices) {
    // 1. Xác định chỉ số luồng (thread index) toàn cục của GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Nếu chỉ số luồng vượt quá số cạnh kề duy nhất, dừng luồng để tránh tràn bộ nhớ
    if (idx >= numUniqueEdges) return;
    
    // 2. Đọc cạnh 64-bit hiện tại và giải mã thành đỉnh nguồn (src) và đỉnh đích (dst)
    uint64_t edge = d_uniqueEdges[idx];
    int src = (int)(edge >> 32);          // Dịch phải 32 bit để lấy ID đỉnh nguồn (32 bit cao)
    int dst = (int)(edge & 0xFFFFFFFF);   // Phép AND bit để lấy ID đỉnh đích (32 bit thấp)
    
    // 3. Lưu đỉnh đích vào mảng danh sách kề. Vị trí lưu trữ trùng khớp với chỉ số cạnh idx
    d_nbrLists[idx] = dst;
    
    // 4. Xử lý phần tử ranh giới đầu tiên
    if (idx == 0) {
        // Cạnh đầu tiên (idx = 0) chắc chắn bắt đầu danh sách kề tại vị trí 0
        d_nbrOffsets[src] = 0;
        
        // Nếu các đỉnh trước src (từ 0 đến src-1) không có cạnh đi ra (đỉnh cô lập),
        // ta gán offset của chúng bằng 0 để số lượng cạnh kề của chúng (offset[i+1] - offset[i]) bằng 0.
        for (int i = 0; i < src; i++) d_nbrOffsets[i] = 0;
    } else {
        // 5. Đối với các cạnh tiếp theo, đọc cạnh đứng ngay trước nó để phát hiện ranh giới thay đổi đỉnh nguồn
        uint64_t prevEdge = d_uniqueEdges[idx - 1];
        int prevSrc = (int)(prevEdge >> 32); // Giải mã đỉnh nguồn của cạnh trước đó
        
        // Nếu phát hiện đỉnh nguồn thay đổi (biên ranh giới giữa 2 cụm đỉnh kề khác nhau)
        if (src != prevSrc) {
            // Danh sách kề của đỉnh src mới bắt đầu tại vị trí idx.
            // Đồng thời gán giá trị idx này cho toàn bộ các đỉnh trống (cô lập) nằm giữa prevSrc và src.
            for (int i = prevSrc + 1; i <= src; i++) {
                d_nbrOffsets[i] = idx;
            }
        }
    }
    
    // 6. Xử lý phần tử cuối cùng để "khóa đuôi" mảng offset
    if (idx == numUniqueEdges - 1) {
        // Đỉnh cuối cùng có cạnh và tất cả các đỉnh cô lập tiếp theo (đến numVertices)
        // đều được gán offset kết thúc là numUniqueEdges (tổng số lượng cạnh kề).
        for (int i = src + 1; i <= numVertices; i++) {
            d_nbrOffsets[i] = numUniqueEdges;
        }
    }
}

// =========================================================================================
// KERNEL: LÀM MƯỢT DỊ HƯỚNG TRÊN GPU (ANISOTROPIC SMOOTHING)
// =========================================================================================
inline __global__ void AnisotropicSmoothingKernel(
    const float3* __restrict__ vertices, const float* __restrict__ sdfIn, float* __restrict__ sdfOut,
    const int* __restrict__ neighborOffsets, const int* __restrict__ neighborLists,
    int numVertices, float sigmaSpatial, float sigmaRange)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    float centerSDF = sdfIn[idx];
    float3 centerPos = vertices[idx];

    float sumWeights = 1.0f;
    float sumSDF = centerSDF * 1.0f;

    int start = neighborOffsets[idx];
    int end = neighborOffsets[idx + 1];

    for (int i = start; i < end; ++i) {
        int nIdx = neighborLists[i];
        float nSDF = sdfIn[nIdx];
        float3 nPos = vertices[nIdx];

        float dx = centerPos.x - nPos.x;
        float dy = centerPos.y - nPos.y;
        float dz = centerPos.z - nPos.z;
        float distSq = dx*dx + dy*dy + dz*dz;

        float valDiff = centerSDF - nSDF;
        float w = expf(-distSq / (2.0f * sigmaSpatial * sigmaSpatial)) *
                  expf(-(valDiff * valDiff) / (2.0f * sigmaRange * sigmaRange));

        sumWeights += w;
        sumSDF += nSDF * w;
    }
    sdfOut[idx] = sumSDF / sumWeights;
}

#endif // SDF_KERNELS_CUH
