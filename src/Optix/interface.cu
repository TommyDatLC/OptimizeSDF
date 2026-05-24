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

inline float* RunOptixConeRayCasting(const OptixGlobalState& state, Matrix<float>& vMat, Matrix<unsigned int>& iMat, Matrix<float>& nMat, int raysPerPoint, float coneAngleRadian) {
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
    OptixTraversableHandle bvhHandle = OptixRunner::BuildBVH(state.context, d_vertex_ptr, d_index_ptr, numVertices, numFaces, d_tempBuffer, d_gasOutputBuffer);
    auto t_bvh_done = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Thời gian Xây dựng BVH: "
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
inline void CaculatingSDFUsingOptix(Model& model, const OptixGlobalState& state, int input_raysPerPoint = 64, float input_angle = 120.0) {
    std::cout << "[DEBUG] [SDF Pipeline] Khởi chạy toàn bộ quy trình tính toán SDF bằng OptiX...\n";
    float coneAngle = input_angle * (3.14159265f / 180.0f);

    Matrix<float>& vertices = model.GetVertexMatrix();
    Matrix<unsigned int>& indices = model.GetVertexIndicesMatrix();
    int numVertices = vertices.Height;
    int numFaces = indices.Height;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Đang cập nhật Vertex Normal cho Model...\n";
    model.UpdateNormal();
    Matrix<float>& vNormal = model.GetVertexNormalMatrix();
    auto t_normal = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Thời gian Cập nhật Normal: " << std::chrono::duration<double>(t_normal - start).count() << " giây\n";


    // 1. TÍNH SDF THÔ TỪ OPTIX
    std::cout << "[DEBUG] [SDF Pipeline] Đang tiến hành OptiX Ray Tracing...\n";
    float* d_rawSDF = RunOptixConeRayCasting(state, vertices, indices, vNormal, input_raysPerPoint, coneAngle);

    float firstRawSDF = 0.0f;
    CUDA_CHECK(cudaMemcpy(&firstRawSDF, d_rawSDF, sizeof(float), cudaMemcpyDeviceToHost));
    auto t_optix_done = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Tính toán OptiX hoàn tất!\n";
    std::cout << "[DEBUG] [SDF Pipeline] Giá trị rawSDF của đỉnh đầu tiên (vừa nhận về từ GPU): " << firstRawSDF << "\n";
    
    // =========================================================================
    // 2. CHUẨN BỊ MẠNG LƯỚI GRAPH CHO VIỆC SMOOTHING TRÊN GPU
    // =========================================================================
    std::cout << "Đóng gói mạng lưới CSR Graph bằng GPU CUB...\n";
    auto t_alloc_csr_start = std::chrono::high_resolution_clock::now();
    int numEdges = numFaces * 6;
    uint64_t *d_edges, *d_sortedEdges;
    CUDA_CHECK(cudaMalloc(&d_edges, numEdges * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sortedEdges, numEdges * sizeof(uint64_t)));
    
    uint64_t *d_uniqueEdges;
    int *d_numUniqueEdges;
    CUDA_CHECK(cudaMalloc(&d_uniqueEdges, numEdges * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_numUniqueEdges, sizeof(int)));

    int *d_nbrOffsets, *d_nbrLists;
    CUDA_CHECK(cudaMalloc((void**)&d_nbrOffsets, (numVertices + 1) * sizeof(int)));
    // We don't know numUniqueEdges yet, so we allocate maximum possible size for d_nbrLists
    CUDA_CHECK(cudaMalloc((void**)&d_nbrLists, numEdges * sizeof(int)));
    
    auto t_alloc_csr_end = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Thời gian Cấp phát bộ nhớ cho CSR Graph (cudaMalloc): " << std::chrono::duration<double>(t_alloc_csr_end - t_alloc_csr_start).count() << " giây\n";

    int blockSize = 256;
    int gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
    GPUGenerateEdges<<<gridSizeFaces, blockSize>>>((const uint3*)indices.getDevicePtr(), numFaces, d_edges);
    CUDA_CHECK(cudaDeviceSynchronize());

    void *d_temp_storage_sort = nullptr;
    size_t temp_storage_bytes_sort = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage_sort, temp_storage_bytes_sort, d_edges, d_sortedEdges, numEdges);
    CUDA_CHECK(cudaMalloc(&d_temp_storage_sort, temp_storage_bytes_sort));
    cub::DeviceRadixSort::SortKeys(d_temp_storage_sort, temp_storage_bytes_sort, d_edges, d_sortedEdges, numEdges);
    CUDA_CHECK(cudaDeviceSynchronize());

    void *d_temp_storage_unique = nullptr;
    size_t temp_storage_bytes_unique = 0;
    cub::DeviceSelect::Unique(d_temp_storage_unique, temp_storage_bytes_unique, d_sortedEdges, d_uniqueEdges, d_numUniqueEdges, numEdges);
    CUDA_CHECK(cudaMalloc(&d_temp_storage_unique, temp_storage_bytes_unique));
    cub::DeviceSelect::Unique(d_temp_storage_unique, temp_storage_bytes_unique, d_sortedEdges, d_uniqueEdges, d_numUniqueEdges, numEdges);
    CUDA_CHECK(cudaDeviceSynchronize());

    int numUniqueEdges = 0;
    CUDA_CHECK(cudaMemcpy(&numUniqueEdges, d_numUniqueEdges, sizeof(int), cudaMemcpyDeviceToHost));

    int gridSizeUnique = (numUniqueEdges + blockSize - 1) / blockSize;
    GPUExtractCSR<<<gridSizeUnique, blockSize>>>(d_uniqueEdges, numUniqueEdges, d_nbrOffsets, d_nbrLists, numVertices);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_csr = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] [SDF Pipeline] Thời gian Xây dựng CSR Graph: " << std::chrono::duration<double>(t_csr - t_optix_done).count() << " giây\n";

    cudaFree(d_edges); cudaFree(d_sortedEdges); cudaFree(d_temp_storage_sort);
    cudaFree(d_uniqueEdges); cudaFree(d_numUniqueEdges); cudaFree(d_temp_storage_unique);

//     =========================================================================
//     3. CHUẨN HÓA VÀ THỰC THI SMOOTHING BẰNG KERNEL TRÊN VRAM CỦA GPU
  //   =========================================================================
    std::cout << "Khởi động Anisotropic Smoothing (GPU)...\n";

//     TÍNH Bounding Box Diagonal và Normalization trên GPU
     auto t_alloc_smooth_start = std::chrono::high_resolution_clock::now();
     float* d_minMaxBox;
     CUDA_CHECK(cudaMalloc((void**)&d_minMaxBox, 6 * sizeof(float)));
     float initBox[6] = {1e15f, -1e15f, 1e15f, -1e15f, 1e15f, -1e15f};
     CUDA_CHECK(cudaMemcpy(d_minMaxBox, initBox, 6 * sizeof(float), cudaMemcpyHostToDevice));

     // Tái sử dụng vùng nhớ GPU đã cấp từ OptiX/CUB pipeline!
     float* d_sdfBuf1 = d_rawSDF; 
     float* d_sdfBuf2;
     cudaMalloc((void**)&d_sdfBuf2, numVertices * sizeof(float));

     float* d_minMaxSDF;
     CUDA_CHECK(cudaMalloc((void**)&d_minMaxSDF, 2 * sizeof(float)));
     float initSDF[2] = {1e15f, -1e15f};
     CUDA_CHECK(cudaMemcpy(d_minMaxSDF, initSDF, 2 * sizeof(float), cudaMemcpyHostToDevice));
     auto t_alloc_smooth_end = std::chrono::high_resolution_clock::now();
     std::cout << "[DEBUG] [SDF Pipeline] Thời gian Cấp phát bộ nhớ cho Smoothing (cudaMalloc + Memcpy): " << std::chrono::duration<double>(t_alloc_smooth_end - t_alloc_smooth_start).count() << " giây\n";

     int gridSizeVerts = (numVertices + blockSize - 1) / blockSize;
     GPUComputeBoundingBox<<<gridSizeVerts, blockSize>>>((const float3*)vertices.getDevicePtr(), numVertices, d_minMaxBox);
     
     float h_minMaxBox[6];
     CUDA_CHECK(cudaMemcpy(h_minMaxBox, d_minMaxBox, 6 * sizeof(float), cudaMemcpyDeviceToHost));
     cudaFree(d_minMaxBox);
     
     float dx = h_minMaxBox[1] - h_minMaxBox[0];
     float dy = h_minMaxBox[3] - h_minMaxBox[2];
     float dz = h_minMaxBox[5] - h_minMaxBox[4];
     float bboxDiagonal = std::sqrt(dx*dx + dy*dy + dz*dz);

     GPUComputeSDFMinMax<<<gridSizeVerts, blockSize>>>(d_sdfBuf1, numVertices, d_minMaxSDF);
     GPUApplySDFNormalization<<<gridSizeVerts, blockSize>>>(d_sdfBuf1, numVertices, d_minMaxSDF);
     CUDA_CHECK(cudaDeviceSynchronize());
     cudaFree(d_minMaxSDF);

   //  COMMENT TEMPORARILY TO TEST rawSDF

     blockSize = 256;
     int gridSize = (numVertices + blockSize - 1) / blockSize;
     int numIterations = 3;
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
    std::cout << "[DEBUG] [SDF Pipeline] Thời gian Làm mượt (Anisotropic Smoothing): " << std::chrono::duration<double>(t_smooth - t_csr).count() << " giây\n";

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
}
#endif