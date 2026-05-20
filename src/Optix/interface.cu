#ifndef SDFMain
#define SDFMain

#include "../Core/Model.cuh"
#include "changeMemoryLayout.cu"
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

// -----------------------------------------------------------------------------
// STRUCT ĐỒNG BỘ VỚI GPU
// -----------------------------------------------------------------------------
struct alignas(8) Params {
    float3* vertices;
    float3* normals;
    float* outputSDF;
    OptixTraversableHandle bvhHandle;
    int raysPerPoint;
    float coneAngleRad;
};

template <typename T>
struct SbtRecord {
    alignas( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
typedef SbtRecord<int> RaygenRecord;
typedef SbtRecord<int> MissRecord;
typedef SbtRecord<int> HitgroupRecord;

#define OPTIX_CHECK( call ) { OptixResult res = call; if( res != OPTIX_SUCCESS ) { fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); exit( 2 ); } }
#define CUDA_CHECK( call ) { cudaError_t error = call; if( error != cudaSuccess ) { fprintf( stderr, "CUDA call (%s) failed with code %d (line %d)\n", #call, error, __LINE__ ); exit( 2 ); } }
static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */) { std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n"; }

// -----------------------------------------------------------------------------
// HÀM CHẠY OPTIX (Đã lược bỏ việc truyền mảng tia từ CPU)
// -----------------------------------------------------------------------------
inline std::vector<float> RunOptixConeRayCasting(Matrix& vMat, Matrix& iMat, Matrix& nMat, int raysPerPoint, float coneAngleRadian, const std::string& ptxCode) {
    int numVertices = vMat.Width;
    int numFaces = iMat.Width;

    std::vector<float3> optixVertices(numVertices);
    std::vector<float3> optixNormals(numVertices);
    std::vector<uint3> optixIndices(numFaces);

    vMat.CopyToHost(); nMat.CopyToHost(); iMat.CopyToHost();
    for(int i=0; i<numVertices; i++) {
        optixVertices[i] = make_float3(vMat.GetHost(0, i), vMat.GetHost(1, i), vMat.GetHost(2, i));
        optixNormals[i]  = make_float3(nMat.GetHost(0, i), nMat.GetHost(1, i), nMat.GetHost(2, i));
    }
    for(int i=0; i<numFaces; i++) {
        optixIndices[i] = make_uint3((uint)iMat.GetHost(0, i), (uint)iMat.GetHost(1, i), (uint)iMat.GetHost(2, i));
    }

    float3 *d_vertices, *d_normals; uint3 *d_indices;
    CUDA_CHECK( cudaMalloc((void**)&d_vertices, numVertices * sizeof(float3)) );
    CUDA_CHECK( cudaMalloc((void**)&d_normals, numVertices * sizeof(float3)) );
    CUDA_CHECK( cudaMalloc((void**)&d_indices, numFaces * sizeof(uint3)) );
    CUDA_CHECK( cudaMemcpy(d_vertices, optixVertices.data(), numVertices * sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_normals, optixNormals.data(), numVertices * sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_indices, optixIndices.data(), numFaces * sizeof(uint3), cudaMemcpyHostToDevice) );

    CUDA_CHECK( cudaFree(0) );
    OptixDeviceContext context = nullptr;
    OptixDeviceContextOptions options = {}; options.logCallbackFunction = &context_log_cb; options.logCallbackLevel = 0;
    OPTIX_CHECK( optixDeviceContextCreate( 0, &options, &context ) );

    CUdeviceptr d_vertex_ptr = (CUdeviceptr)d_vertices;
    OptixBuildInput triangleInput = {};
    triangleInput.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangleInput.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.numVertices   = numVertices;
    triangleInput.triangleArray.vertexBuffers = &d_vertex_ptr;
    triangleInput.triangleArray.indexFormat   = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.numIndexTriplets = numFaces;
    triangleInput.triangleArray.indexBuffer   = (CUdeviceptr)d_indices;

    uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING };
    triangleInput.triangleArray.flags         = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accelOptions, &triangleInput, 1, &gasBufferSizes ) );

    CUdeviceptr d_tempBuffer, d_gasOutputBuffer;
    CUDA_CHECK( cudaMalloc((void**)&d_tempBuffer, gasBufferSizes.tempSizeInBytes) );
    CUDA_CHECK( cudaMalloc((void**)&d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes) );

    OptixTraversableHandle bvhHandle = 0;
    OPTIX_CHECK( optixAccelBuild( context, 0, &accelOptions, &triangleInput, 1, d_tempBuffer, gasBufferSizes.tempSizeInBytes, d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes, &bvhHandle, nullptr, 0 ) );
    CUDA_CHECK( cudaFree((void*)d_tempBuffer) );

    OptixModule module = nullptr;
    OptixModuleCompileOptions moduleCompileOptions = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur        = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues      = 1;
    pipelineCompileOptions.numAttributeValues    = 2;
    pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    OPTIX_CHECK( optixModuleCreate( context, &moduleCompileOptions, &pipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), nullptr, nullptr, &module ) );

    OptixProgramGroup raygenProgGroup, missProgGroup, hitProgGroup;
    OptixProgramGroupOptions pgOptions = {};

    OptixProgramGroupDesc raygenDesc = {}; raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; raygenDesc.raygen.module = module; raygenDesc.raygen.entryFunctionName = "__raygen__sdf_cone";
    OPTIX_CHECK( optixProgramGroupCreate( context, &raygenDesc, 1, &pgOptions, nullptr, nullptr, &raygenProgGroup ) );

    OptixProgramGroupDesc missDesc = {}; missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    OPTIX_CHECK( optixProgramGroupCreate( context, &missDesc, 1, &pgOptions, nullptr, nullptr, &missProgGroup ) );

    OptixProgramGroupDesc hitDesc = {}; hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; hitDesc.hitgroup.moduleCH = module; hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__sdf";
    OPTIX_CHECK( optixProgramGroupCreate( context, &hitDesc, 1, &pgOptions, nullptr, nullptr, &hitProgGroup ) );

    OptixProgramGroup programGroups[] = { raygenProgGroup, missProgGroup, hitProgGroup };
    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions pipelineLinkOptions = {}; pipelineLinkOptions.maxTraceDepth = 1;
    OPTIX_CHECK( optixPipelineCreate( context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups, 3, nullptr, nullptr, &pipeline ) );

    RaygenRecord rgSbt; OPTIX_CHECK( optixSbtRecordPackHeader( raygenProgGroup, &rgSbt ) ); CUdeviceptr d_rgSbt; CUDA_CHECK( cudaMalloc((void**)&d_rgSbt, sizeof(RaygenRecord)) ); CUDA_CHECK( cudaMemcpy((void*)d_rgSbt, &rgSbt, sizeof(RaygenRecord), cudaMemcpyHostToDevice) );
    MissRecord msSbt; OPTIX_CHECK( optixSbtRecordPackHeader( missProgGroup, &msSbt ) ); CUdeviceptr d_msSbt; CUDA_CHECK( cudaMalloc((void**)&d_msSbt, sizeof(MissRecord)) ); CUDA_CHECK( cudaMemcpy((void*)d_msSbt, &msSbt, sizeof(MissRecord), cudaMemcpyHostToDevice) );
    HitgroupRecord hgSbt; OPTIX_CHECK( optixSbtRecordPackHeader( hitProgGroup, &hgSbt ) ); CUdeviceptr d_hgSbt; CUDA_CHECK( cudaMalloc((void**)&d_hgSbt, sizeof(HitgroupRecord)) ); CUDA_CHECK( cudaMemcpy((void*)d_hgSbt, &hgSbt, sizeof(HitgroupRecord), cudaMemcpyHostToDevice) );

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = d_rgSbt; sbt.missRecordBase = d_msSbt; sbt.missRecordStrideInBytes = sizeof( MissRecord ); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = d_hgSbt; sbt.hitgroupRecordStrideInBytes = sizeof( HitgroupRecord ); sbt.hitgroupRecordCount = 1;

    float* d_outputSDF;
    CUDA_CHECK( cudaMalloc((void**)&d_outputSDF, numVertices * sizeof(float)) );
    CUDA_CHECK( cudaMemset(d_outputSDF, 0, numVertices * sizeof(float)) );

    Params params = {};
    params.vertices     = d_vertices;
    params.normals      = d_normals;
    params.outputSDF    = d_outputSDF;
    params.bvhHandle    = bvhHandle;
    params.raysPerPoint = raysPerPoint;
    params.coneAngleRad = coneAngleRadian;

    CUdeviceptr d_params;
    CUDA_CHECK( cudaMalloc((void**)&d_params, sizeof(Params)) );
    CUDA_CHECK( cudaMemcpy((void*)d_params, &params, sizeof(Params), cudaMemcpyHostToDevice) );

    std::cout << "OptiX: Đang bắn " << raysPerPoint << " tia Hammersley Uniform/đỉnh...\n";
    OPTIX_CHECK( optixLaunch( pipeline, 0, d_params, sizeof(Params), &sbt, numVertices, 1, 1 ) );
    CUDA_CHECK( cudaDeviceSynchronize() );

    std::vector<float> results(numVertices);
    CUDA_CHECK( cudaMemcpy(results.data(), d_outputSDF, numVertices * sizeof(float), cudaMemcpyDeviceToHost) );

    cudaFree(d_vertices); cudaFree(d_normals); cudaFree(d_indices);
    cudaFree((void*)d_gasOutputBuffer); cudaFree((void*)d_rgSbt);
    cudaFree((void*)d_msSbt); cudaFree((void*)d_hgSbt);
    cudaFree((void*)d_params); cudaFree(d_outputSDF);

    optixPipelineDestroy(pipeline); optixProgramGroupDestroy(raygenProgGroup); optixProgramGroupDestroy(missProgGroup); optixProgramGroupDestroy(hitProgGroup); optixModuleDestroy(module); optixDeviceContextDestroy(context);

    return results;
}

// -----------------------------------------------------------------------------
// HÀM CHÍNH GỌI TỪ NGOÀI VÀO (OPTIX + LÀM MƯỢT TRÊN GPU)
// -----------------------------------------------------------------------------
inline void CaculatingSDFUsingOptix(Model& model, const std::string& shader, int input_raysPerPoint = 64, float input_angle = 150.0) {
    std::cout << "Bắt đầu tính toán SDF bằng OptiX...\n";
    float coneAngle = input_angle * (3.14159265f / 180.0f);

    Matrix& vertices = model.GetVertexMatrix();
    Matrix& indices = model.GetVertexIndicesMatrix();
    int numVertices = vertices.Width;
    int numFaces = indices.Width;

    auto start = std::chrono::high_resolution_clock::now();
    model.UpdateNormal();
    Matrix& vNormal = model.GetVertexNormalMatrix();

    // 1. TÍNH SDF THÔ TỪ OPTIX (ĐÃ CÓ TRỌNG SỐ VÀ OUTLIER TỪ KERNEL)
    std::vector<float> rawSDF = RunOptixConeRayCasting(vertices, indices, vNormal, input_raysPerPoint, coneAngle, shader);

    // =========================================================================
    // 2. CHUẨN BỊ MẠNG LƯỚI GRAPH CHO VIỆC SMOOTHING TRÊN GPU
    // =========================================================================
    std::cout << "Đóng gói mạng lưới CSR Graph...\n";
    std::vector<std::vector<int>> adjacency(numVertices);
    indices.CopyToHost();
    for (int i = 0; i < numFaces; i++) {
        int v0 = (int)indices.GetHost(0, i), v1 = (int)indices.GetHost(1, i), v2 = (int)indices.GetHost(2, i);
        adjacency[v0].push_back(v1); adjacency[v0].push_back(v2);
        adjacency[v1].push_back(v0); adjacency[v1].push_back(v2);
        adjacency[v2].push_back(v0); adjacency[v2].push_back(v1);
    }

    std::vector<int> h_neighborOffsets(numVertices + 1, 0);
    std::vector<int> h_neighborLists;
    for (int i = 0; i < numVertices; i++) {
        std::sort(adjacency[i].begin(), adjacency[i].end());
        adjacency[i].erase(std::unique(adjacency[i].begin(), adjacency[i].end()), adjacency[i].end());
        h_neighborOffsets[i] = h_neighborLists.size();
        h_neighborLists.insert(h_neighborLists.end(), adjacency[i].begin(), adjacency[i].end());
    }
    h_neighborOffsets[numVertices] = h_neighborLists.size();

    // =========================================================================
    // 3. THỰC THI SMOOTHING BẰNG KERNEL TRÊN VRAM CỦA GPU
    // =========================================================================
    std::cout << "Khởi động Anisotropic Smoothing (GPU)...\n";
    float3* d_vertices; float* d_sdfBuf1; float* d_sdfBuf2; int* d_nbrOffsets; int* d_nbrLists;

    cudaMalloc((void**)&d_vertices, numVertices * sizeof(float3));
    cudaMalloc((void**)&d_sdfBuf1, numVertices * sizeof(float));
    cudaMalloc((void**)&d_sdfBuf2, numVertices * sizeof(float));
    cudaMalloc((void**)&d_nbrOffsets, (numVertices + 1) * sizeof(int));
    cudaMalloc((void**)&d_nbrLists, h_neighborLists.size() * sizeof(int));

    int blockSize = 256;
    int gridSize = (numVertices + blockSize - 1) / blockSize;
    ConvertMatrixToFloat3<<<gridSize, blockSize>>>(vertices.getDevicePtr(), d_vertices, numVertices, vertices.Height);

    cudaMemcpy(d_sdfBuf1, rawSDF.data(), numVertices * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nbrOffsets, h_neighborOffsets.data(), (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nbrLists, h_neighborLists.data(), h_neighborLists.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Thuật toán chạy 3 vòng lặp Ping-Pong đảo Buffer qua lại
    int numIterations = 3;
    float sigmaSpatial = 0.05f;
    float sigmaRange = 0.1f;

    for (int iter = 0; iter < numIterations; iter++) {
        float* d_in = (iter % 2 == 0) ? d_sdfBuf1 : d_sdfBuf2;
        float* d_out = (iter % 2 == 0) ? d_sdfBuf2 : d_sdfBuf1;

        AnisotropicSmoothingKernel<<<gridSize, blockSize>>>(
            d_vertices, d_in, d_out, d_nbrOffsets, d_nbrLists,
            numVertices, sigmaSpatial, sigmaRange
        );
        cudaDeviceSynchronize();
    }

    std::vector<float> finalSDF(numVertices);
    float* d_finalOut = (numIterations % 2 == 0) ? d_sdfBuf1 : d_sdfBuf2;
    cudaMemcpy(finalSDF.data(), d_finalOut, numVertices * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vertices); cudaFree(d_sdfBuf1); cudaFree(d_sdfBuf2);
    cudaFree(d_nbrOffsets); cudaFree(d_nbrLists);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;
    std::cout << "Thời gian tổng cộng (OptiX + GPU Smooth): " << duration.count() << " giây\n";

    double maxDist = 0.0;
    for (int i = 0; i < vertices.Width; ++i) {
        if (finalSDF[i] > maxDist) maxDist = finalSDF[i];
        model.AddHeatMapVertexForPreviewEngine(i, finalSDF[i]);
    }

    model.AddToScene("Optix_SDF_Model_Smoothed", false);
}
#endif