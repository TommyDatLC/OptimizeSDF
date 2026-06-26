#ifndef SDFMain
#define SDFMain

#include "../Core/Model.cuh"
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cuda.h>
#include "../../Core/Helper.hpp"
#include "OptixHostUtils.cuh"
#include "SDFKernels.cuh"

// -----------------------------------------------------------------------------
// HÀM CHẠY OPTIX (ZERO-COPY: KHÔNG CẦN CHUYỂN ĐỔI BỘ NHỚ!)
// -----------------------------------------------------------------------------
#include "OptixRunner.cuh"

struct OptixGlobalState {
    OptixDeviceContext context;
    OptixModule module;
    OptixProgramGroup raygenProgGroup;
    OptixProgramGroup missProgGroup;
    OptixProgramGroup hitProgGroup;
    OptixPipeline pipeline;
    CUdeviceptr d_rgSbt;
    CUdeviceptr d_msSbt;
    CUdeviceptr d_hgSbt;
    OptixShaderBindingTable sbt;
};

inline OptixGlobalState InitializeOptixGlobalState(const std::string& ptxCode) {
    auto t_init_start = std::chrono::high_resolution_clock::now();
    OptixGlobalState state;
    state.context = OptixRunner::InitContext();
    state.module = nullptr;
    state.pipeline = OptixRunner::CreatePipeline(state.context, ptxCode, state.raygenProgGroup, state.missProgGroup, state.hitProgGroup, state.module);
    state.sbt = OptixRunner::BuildSBT(state.raygenProgGroup, state.missProgGroup, state.hitProgGroup, state.d_rgSbt, state.d_msSbt, state.d_hgSbt);

    auto t_init_done = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Thời gian Khởi tạo (Context, JIT Compile PTX, SBT): "
              << std::chrono::duration<double>(t_init_done - t_init_start).count() << " giây\n";
    return state;
}

inline void DestroyOptixGlobalState(OptixGlobalState& state) {
    CUDA_CHECK( cudaFree((void*)state.d_rgSbt) );
    CUDA_CHECK( cudaFree((void*)state.d_msSbt) );
    CUDA_CHECK( cudaFree((void*)state.d_hgSbt) );
    optixPipelineDestroy(state.pipeline);
    optixProgramGroupDestroy(state.raygenProgGroup);
    optixProgramGroupDestroy(state.missProgGroup);
    optixProgramGroupDestroy(state.hitProgGroup);
    optixModuleDestroy(state.module);
    optixDeviceContextDestroy(state.context);
}

inline float* RunOptixConeRayCasting(const OptixGlobalState& state, Matrix<float>& vMat, Matrix<unsigned int>& iMat, Matrix<float>& nMat, int raysPerPoint, float coneAngleRadian, cudaStream_t streamBVH, cudaStream_t streamNormal) {
    std::cout << "[DEBUG] [SDF Pipeline] Bắt đầu khởi tạo OptiX Pipeline và Build BVH...\n";
    int numVertices = vMat.Height;
    int numFaces = iMat.Height;
    std::cout << "[DEBUG] [SDF Pipeline] Cấu hình tia: " << raysPerPoint << " tia/đỉnh | Góc nón: " << coneAngleRadian << " rad\n";
    std::cout << "[DEBUG] [SDF Pipeline] Tổng số đỉnh: " << numVertices << " | Tổng số mặt: " << numFaces << "\n";

    CUdeviceptr d_vertex_ptr = (CUdeviceptr)vMat.getDevicePtr();
    CUdeviceptr d_index_ptr = (CUdeviceptr)iMat.getDevicePtr();
    float3* d_normals = (float3*)nMat.getDevicePtr();

    auto t_bvh_start = std::chrono::high_resolution_clock::now();
    CUdeviceptr d_tempBuffer, d_gasOutputBuffer;
    OptixTraversableHandle bvhHandle = OptixRunner::BuildBVH(state.context, d_vertex_ptr, d_index_ptr, numVertices, numFaces, d_tempBuffer, d_gasOutputBuffer, streamBVH);
    
    // Đồng bộ cả 2 luồng trước khi chạy OptiX Ray Tracing vì Ray Tracing cần cả Cây BVH và Normal
    cudaStreamSynchronize(streamBVH);
    cudaStreamSynchronize(streamNormal);
    
    auto t_bvh_done = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Thời gian Chuẩn bị BVH + Normal (Chạy Song Song): "
              << std::chrono::duration<double>(t_bvh_done - t_bvh_start).count() << " giây\n";

    float* d_rawSDF = OptixRunner::LaunchOptixAndCUB(
        numVertices, raysPerPoint, coneAngleRadian,
        d_vertex_ptr, d_normals,
        bvhHandle, state.pipeline, state.sbt
    );

    CUDA_CHECK( cudaFree((void*)d_tempBuffer) );
    CUDA_CHECK( cudaFree((void*)d_gasOutputBuffer) );

    return d_rawSDF;
}

// -----------------------------------------------------------------------------
// HÀM CHÍNH GỌI TỪ NGOÀI VÀO (OPTIX + LÀM MƯỢT TRÊN GPU)
// -----------------------------------------------------------------------------
inline float CaculatingSDFUsingOptix(Model& model, const OptixGlobalState& state, int input_raysPerPoint = 64, float input_angle = 120.0) {
    std::cout << "[DEBUG] [SDF Pipeline] Khởi chạy toàn bộ quy trình tính toán SDF bằng OptiX...\n";
    float coneAngle = input_angle * (3.14159265f / 180.0f);

    Matrix<float>& vertices = model.GetVertexMatrix();
    Matrix<unsigned int>& indices = model.GetVertexIndicesMatrix();
    int numVertices = vertices.Height;
    int numFaces = indices.Height;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Đang cập nhật Vertex Normal và Build BVH (Song song bằng CUDA Streams)...\n";
    
    cudaStream_t streamNormal, streamBVH;
    cudaStreamCreate(&streamNormal);
    cudaStreamCreate(&streamBVH);

    model.UpdateNormal(streamNormal);
    Matrix<float>& vNormal = model.GetVertexNormalMatrix();

    // 1. TÍNH SDF THÔ TỪ OPTIX (Bao gồm cả chờ Build BVH)
    std::cout << "[DEBUG] [SDF Pipeline] Đang tiến hành OptiX Ray Tracing...\n";
    float* d_rawSDF = RunOptixConeRayCasting(state, vertices, indices, vNormal, input_raysPerPoint, coneAngle, streamBVH, streamNormal);
    
    cudaStreamDestroy(streamNormal);
    cudaStreamDestroy(streamBVH);

    float firstRawSDF = 0.0f;
    CUDA_CHECK(cudaMemcpy(&firstRawSDF, d_rawSDF, sizeof(float), cudaMemcpyDeviceToHost));
    auto t_optix_done = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Tính toán OptiX hoàn tất!\n";
    
    // =========================================================================
    // 2. CHUẨN BỊ MẠNG LƯỚI GRAPH VÀ NORMALIZATION SONG SONG
    // =========================================================================
    std::cout << "Đóng gói mạng lưới CSR Graph và Chuẩn hóa SDF song song...\n";
    
    cudaStream_t streamCSR, streamNorm;
    CUDA_CHECK(cudaStreamCreate(&streamCSR));
    CUDA_CHECK(cudaStreamCreate(&streamNorm));

    auto t_alloc_start = std::chrono::high_resolution_clock::now();
    int numEdges = numFaces * 6;
    uint64_t *d_edges, *d_sortedEdges, *d_uniqueEdges;
    int *d_numUniqueEdges, *d_nbrOffsets, *d_nbrLists;
    CUDA_CHECK(cudaMalloc(&d_edges, numEdges * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sortedEdges, numEdges * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_uniqueEdges, numEdges * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_numUniqueEdges, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_nbrOffsets, (numVertices + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_nbrLists, numEdges * sizeof(int)));

    float* d_minMaxBox;
    CUDA_CHECK(cudaMalloc((void**)&d_minMaxBox, 6 * sizeof(float)));
    float initBox[6] = {1e15f, -1e15f, 1e15f, -1e15f, 1e15f, -1e15f};
    CUDA_CHECK(cudaMemcpyAsync(d_minMaxBox, initBox, 6 * sizeof(float), cudaMemcpyHostToDevice, streamNorm));

    float* d_sdfBuf1 = d_rawSDF; 
    float* d_sdfBuf2;
    CUDA_CHECK(cudaMalloc((void**)&d_sdfBuf2, numVertices * sizeof(float)));

    float* d_minMaxSDF;
    CUDA_CHECK(cudaMalloc((void**)&d_minMaxSDF, 2 * sizeof(float)));
    float initSDF[2] = {1e15f, -1e15f};
    CUDA_CHECK(cudaMemcpyAsync(d_minMaxSDF, initSDF, 2 * sizeof(float), cudaMemcpyHostToDevice, streamNorm));

    auto t_alloc_end = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Thời gian Cấp phát bộ nhớ chung (cudaMalloc): " << std::chrono::duration<double>(t_alloc_end - t_alloc_start).count() << " giây\n";

    // --- THỰC THI CSR (streamCSR) ---
    int blockSize = 256;
    int gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
    GPUGenerateEdges<<<gridSizeFaces, blockSize, 0, streamCSR>>>((const uint3*)indices.getDevicePtr(), numFaces, d_edges);

    void *d_temp_storage_sort = nullptr; size_t temp_storage_bytes_sort = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage_sort, temp_storage_bytes_sort, d_edges, d_sortedEdges, numEdges, 0, sizeof(uint64_t)*8, streamCSR);
    CUDA_CHECK(cudaMalloc(&d_temp_storage_sort, temp_storage_bytes_sort));
    cub::DeviceRadixSort::SortKeys(d_temp_storage_sort, temp_storage_bytes_sort, d_edges, d_sortedEdges, numEdges, 0, sizeof(uint64_t)*8, streamCSR);

    void *d_temp_storage_unique = nullptr; size_t temp_storage_bytes_unique = 0;
    cub::DeviceSelect::Unique(d_temp_storage_unique, temp_storage_bytes_unique, d_sortedEdges, d_uniqueEdges, d_numUniqueEdges, numEdges, streamCSR);
    CUDA_CHECK(cudaMalloc(&d_temp_storage_unique, temp_storage_bytes_unique));
    cub::DeviceSelect::Unique(d_temp_storage_unique, temp_storage_bytes_unique, d_sortedEdges, d_uniqueEdges, d_numUniqueEdges, numEdges, streamCSR);

    int numUniqueEdges = 0;
    CUDA_CHECK(cudaMemcpyAsync(&numUniqueEdges, d_numUniqueEdges, sizeof(int), cudaMemcpyDeviceToHost, streamCSR));

    // --- THỰC THI NORMALIZATION (streamNorm) ---
    int gridSizeVerts = (numVertices + blockSize - 1) / blockSize;
    GPUComputeBoundingBox<<<gridSizeVerts, blockSize, 0, streamNorm>>>((const float3*)vertices.getDevicePtr(), numVertices, d_minMaxBox);
    
    float h_minMaxBox[6];
    CUDA_CHECK(cudaMemcpyAsync(h_minMaxBox, d_minMaxBox, 6 * sizeof(float), cudaMemcpyDeviceToHost, streamNorm));

    GPUComputeSDFMinMax<<<gridSizeVerts, blockSize, 0, streamNorm>>>(d_sdfBuf1, numVertices, d_minMaxSDF);
    GPUApplySDFNormalization<<<gridSizeVerts, blockSize, 0, streamNorm>>>(d_sdfBuf1, numVertices, d_minMaxSDF);

    // --- ĐỒNG BỘ HAI STREAM ---
    CUDA_CHECK(cudaStreamSynchronize(streamCSR));
    CUDA_CHECK(cudaStreamSynchronize(streamNorm));

    // Tiếp tục hoàn tất CSR sau khi có numUniqueEdges
    int gridSizeUnique = (numUniqueEdges + blockSize - 1) / blockSize;
    GPUExtractCSR<<<gridSizeUnique, blockSize, 0, streamCSR>>>(d_uniqueEdges, numUniqueEdges, d_nbrOffsets, d_nbrLists, numVertices);
    CUDA_CHECK(cudaStreamSynchronize(streamCSR));

    auto t_parallel_done = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Thời gian thực thi CSR & Normalization song song: " << std::chrono::duration<double>(t_parallel_done - t_optix_done).count() << " giây\n";

    cudaFree(d_edges); cudaFree(d_sortedEdges); cudaFree(d_temp_storage_sort);
    cudaFree(d_uniqueEdges); cudaFree(d_numUniqueEdges); cudaFree(d_temp_storage_unique);
    cudaFree(d_minMaxBox); cudaFree(d_minMaxSDF);

    CUDA_CHECK(cudaStreamDestroy(streamCSR));
    CUDA_CHECK(cudaStreamDestroy(streamNorm));

//     =========================================================================
//     3. THỰC THI SMOOTHING BẰNG KERNEL TRÊN VRAM CỦA GPU
//     =========================================================================
    std::cout << "Khởi động Anisotropic Smoothing (GPU)...\n";
    float dx = h_minMaxBox[1] - h_minMaxBox[0];
    float dy = h_minMaxBox[3] - h_minMaxBox[2];
    float dz = h_minMaxBox[5] - h_minMaxBox[4];
    float bboxDiagonal = std::sqrt(dx*dx + dy*dy + dz*dz);

   //  COMMENT TEMPORARILY TO TEST rawSDF

     blockSize = 256;
     int gridSize = (numVertices + blockSize - 1) / blockSize;
     int numIterations = 3; // Đã bật lại smoothing với 3 iterations theo yêu cầu
     float sigmaSpatial = bboxDiagonal * 0.02f; // Tính sigma không gian động dựa trên bounding box
     float sigmaRange = 0.1f; // sigmaRange = 0.1f chuẩn hóa cho range [0, 1]

     // Tái sử dụng trực tiếp con trỏ Matrix, không cần mảng temp d_vertices
     float3* d_vertices_direct = (float3*)vertices.getDevicePtr();

     for (int iter = 0; iter < numIterations; iter++) {
         float* d_in = (iter % 2 == 0) ? d_sdfBuf1 : d_sdfBuf2;
         float* d_out = (iter % 2 == 0) ? d_sdfBuf2 : d_sdfBuf1;

         AnisotropicSmoothingKernel<<<gridSize, blockSize>>>(
             d_vertices_direct, d_in, d_out, d_nbrOffsets, d_nbrLists,
             numVertices, sigmaSpatial, sigmaRange
         );
         cudaDeviceSynchronize();
     }
    auto t_smooth = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Thời gian Làm mượt (Anisotropic Smoothing): " << std::chrono::duration<double>(t_smooth - t_parallel_done).count() << " giây\n";

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;
    std::cout << "Thời gian tổng cộng (OptiX + GPU Smooth): " << duration.count() << " giây\n";
     std::vector<float> finalSDF(numVertices);
     float* d_finalOut = (numIterations % 2 == 0) ? d_sdfBuf1 : d_sdfBuf2;
     cudaMemcpy(finalSDF.data(), d_finalOut, numVertices * sizeof(float), cudaMemcpyDeviceToHost);

     cudaFree(d_sdfBuf1); cudaFree(d_sdfBuf2);
     cudaFree(d_nbrOffsets); cudaFree(d_nbrLists);

    for (int i = 0; i < vertices.Height; ++i) {
        model.AddHeatMapVertexForPreviewEngine(i, finalSDF[i]);
    }
    model.SetShowHeatMap(true);
    model.AddToScene("Optix_SDF_Model_Smoothed", true);
    return duration.count();
}
#endif