#ifndef SDFMain
#define SDFMain

#include "../Core/Model.cuh"
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <cmath> // Thêm thư viện toán học
#include <algorithm> // Thêm để dùng std::max

// Bắt buộc phải có định nghĩa này để khởi tạo các hàm của OptiX
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <chrono>

#include "../../Core/Helper.hpp"

// -----------------------------------------------------------------------------
// STRUCT ĐỒNG BỘ VỚI GPU (Đã thêm padding để khớp bộ nhớ 48 bytes)
// -----------------------------------------------------------------------------
struct alignas(8) Params {
    float3* vertices;
    float3* normals;
    float* outputSDF;
    float3* localRays;
    OptixTraversableHandle bvhHandle;
    int raysPerPoint;
    int padding; // Đệm cho chẵn bộ nhớ
};

template <typename T>
struct SbtRecord {
    alignas( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<int> RaygenRecord;
typedef SbtRecord<int> MissRecord;
typedef SbtRecord<int> HitgroupRecord;

#define OPTIX_CHECK( call )                                                    \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS ) {                                           \
            fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
            exit( 2 );                                                         \
        }                                                                      \
    }

#define CUDA_CHECK( call )                                                     \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess ) {                                           \
            fprintf( stderr, "CUDA call (%s) failed with code %d (line %d)\n", #call, error, __LINE__ ); \
            exit( 2 );                                                         \
        }                                                                      \
    }

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

// -----------------------------------------------------------------------------
// HÀM CHÍNH THỰC THI OPTIX PIPELINE
// -----------------------------------------------------------------------------
inline std::vector<float> RunOptixConeRayCasting(Matrix& vMat, Matrix& iMat, Matrix& nMat, int raysPerPoint, float coneAngleRadian, const std::string& ptxCode ) {
    int numVertices = vMat.Width;
    int numFaces = iMat.Width;

    // =========================================================================
    // BƯỚC 0: TÍNH TOÁN SPHERICAL FIBONACCI TRÊN CPU MỘT LẦN DUY NHẤT
    // =========================================================================
    std::vector<float3> h_localRays(raysPerPoint);
    float z_min = std::cos(coneAngleRadian / 2.0f);
    float goldenRatio = 1.61803398875f;

    for (int i = 0; i < raysPerPoint; ++i) {
        float z = 1.0f - (1.0f - z_min) * ((float)i / (float)(raysPerPoint - 1));
        float phi = 2.0f * 3.14159265359f * ((float)i / goldenRatio);
        float radius = std::sqrt(std::max(0.0f, 1.0f - z * z));

        h_localRays[i] = make_float3(radius * std::cos(phi), radius * std::sin(phi), z);
    }

    // =========================================================================
    // BƯỚC 1: CHUẨN BỊ DỮ LIỆU ĐÚNG ĐỊNH DẠNG CHO OPTIX
    // =========================================================================
    std::vector<float3> optixVertices(numVertices);
    std::vector<float3> optixNormals(numVertices);
    std::vector<uint3> optixIndices(numFaces);

    vMat.CopyToHost();
    nMat.CopyToHost();
    iMat.CopyToHost();
    for(int i=0; i<numVertices; i++) {
        optixVertices[i] = make_float3(vMat.GetHost(0, i), vMat.GetHost(1, i), vMat.GetHost(2, i));
        optixNormals[i]  = make_float3(nMat.GetHost(0, i), nMat.GetHost(1, i), nMat.GetHost(2, i));
    }
    for(int i=0; i<numFaces; i++) {
        optixIndices[i] = make_uint3((uint)iMat.GetHost(0, i), (uint)iMat.GetHost(1, i), (uint)iMat.GetHost(2, i));
    }

    float3 *d_vertices, *d_normals, *d_localRays;
    uint3 *d_indices;
    CUDA_CHECK( cudaMalloc((void**)&d_vertices, numVertices * sizeof(float3)) );
    CUDA_CHECK( cudaMalloc((void**)&d_normals, numVertices * sizeof(float3)) );
    // ĐÃ SỬA: Sửa thành sizeof(uint3) thay vì sizeof(uint)
    CUDA_CHECK( cudaMalloc((void**)&d_indices, numFaces * sizeof(uint3)) );
    CUDA_CHECK( cudaMalloc((void**)&d_localRays, raysPerPoint * sizeof(float3)) );

    CUDA_CHECK( cudaMemcpy(d_vertices, optixVertices.data(), numVertices * sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_normals, optixNormals.data(), numVertices * sizeof(float3), cudaMemcpyHostToDevice) );
    // ĐÃ SỬA: Sửa thành sizeof(uint3) thay vì sizeof(uint)
    CUDA_CHECK( cudaMemcpy(d_indices, optixIndices.data(), numFaces * sizeof(uint3), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_localRays, h_localRays.data(), raysPerPoint * sizeof(float3), cudaMemcpyHostToDevice) );

    // =========================================================================
    // KHỞI TẠO OPTIX CONTEXT
    // =========================================================================
    CUDA_CHECK( cudaFree(0) );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 0;

    OptixDeviceContext context = nullptr;
    OPTIX_CHECK( optixDeviceContextCreate( 0, &options, &context ) );

    // =========================================================================
    // 1. TẠO BVH (GEOMETRY ACCELERATION STRUCTURE - GAS)
    // =========================================================================
    // ĐÃ SỬA: Gán con trỏ an toàn để truyền mảng vertex vào OptiX
    CUdeviceptr d_vertex_ptr = (CUdeviceptr)d_vertices;

    OptixBuildInput triangleInput = {};
    triangleInput.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangleInput.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.numVertices   = numVertices;
    triangleInput.triangleArray.vertexBuffers = &d_vertex_ptr;
    triangleInput.triangleArray.indexFormat   = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.numIndexTriplets = numFaces;
    triangleInput.triangleArray.indexBuffer   = (CUdeviceptr)d_indices;

    uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE  };
    triangleInput.triangleArray.flags         = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accelOptions, &triangleInput, 1, &gasBufferSizes ) );

    CUdeviceptr d_tempBuffer, d_gasOutputBuffer;
    CUDA_CHECK( cudaMalloc((void**)&d_tempBuffer, gasBufferSizes.tempSizeInBytes) );
    CUDA_CHECK( cudaMalloc((void**)&d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes) );

    OptixTraversableHandle bvhHandle = 0;
    OPTIX_CHECK( optixAccelBuild( context, 0, &accelOptions, &triangleInput, 1, d_tempBuffer, gasBufferSizes.tempSizeInBytes, d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes, &bvhHandle, nullptr, 0 ) );

    // Giờ lệnh cudaFree này sẽ an toàn vì optixAccelBuild bên trên không còn bị văng lỗi VRAM nữa
    CUDA_CHECK( cudaFree((void*)d_tempBuffer) );

    // =========================================================================
    // 2. TẠO MODULE (TẢI FILE PTX) VÀ PIPELINE
    // =========================================================================
    OptixModuleCompileOptions moduleCompileOptions = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur        = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues      = 1;
    pipelineCompileOptions.numAttributeValues    = 2;
    pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    OptixModule module = nullptr;
    OPTIX_CHECK( optixModuleCreate( context, &moduleCompileOptions, &pipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), nullptr, nullptr, &module ) );

    OptixProgramGroup raygenProgGroup, missProgGroup, hitProgGroup;
    OptixProgramGroupOptions pgOptions = {};

    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module            = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__sdf_cone";
    OPTIX_CHECK( optixProgramGroupCreate( context, &raygenDesc, 1, &pgOptions, nullptr, nullptr, &raygenProgGroup ) );

    OptixProgramGroupDesc missDesc = {};
    missDesc.kind   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    OPTIX_CHECK( optixProgramGroupCreate( context, &missDesc, 1, &pgOptions, nullptr, nullptr, &missProgGroup ) );

    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH            = module;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__sdf";
    OPTIX_CHECK( optixProgramGroupCreate( context, &hitDesc, 1, &pgOptions, nullptr, nullptr, &hitProgGroup ) );

    OptixProgramGroup programGroups[] = { raygenProgGroup, missProgGroup, hitProgGroup };

    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;
    OPTIX_CHECK( optixPipelineCreate( context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups, 3, nullptr, nullptr, &pipeline ) );

    // =========================================================================
    // 3. CẤU HÌNH SHADER BINDING TABLE (SBT)
    // =========================================================================
    RaygenRecord rgSbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( raygenProgGroup, &rgSbt ) );
    CUdeviceptr d_rgSbt;
    CUDA_CHECK( cudaMalloc((void**)&d_rgSbt, sizeof(RaygenRecord)) );
    CUDA_CHECK( cudaMemcpy((void*)d_rgSbt, &rgSbt, sizeof(RaygenRecord), cudaMemcpyHostToDevice) );

    MissRecord msSbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( missProgGroup, &msSbt ) );
    CUdeviceptr d_msSbt;
    CUDA_CHECK( cudaMalloc((void**)&d_msSbt, sizeof(MissRecord)) );
    CUDA_CHECK( cudaMemcpy((void*)d_msSbt, &msSbt, sizeof(MissRecord), cudaMemcpyHostToDevice) );

    HitgroupRecord hgSbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( hitProgGroup, &hgSbt ) );
    CUdeviceptr d_hgSbt;
    CUDA_CHECK( cudaMalloc((void**)&d_hgSbt, sizeof(HitgroupRecord)) );
    CUDA_CHECK( cudaMemcpy((void*)d_hgSbt, &hgSbt, sizeof(HitgroupRecord), cudaMemcpyHostToDevice) );

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord                = d_rgSbt;
    sbt.missRecordBase              = d_msSbt;
    sbt.missRecordStrideInBytes     = sizeof( MissRecord );
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = d_hgSbt;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitgroupRecord );
    sbt.hitgroupRecordCount         = 1;

    // =========================================================================
    // 4. CHUẨN BỊ PARAMS VÀ OPTIX LAUNCH
    // =========================================================================
    float* d_outputSDF;
    CUDA_CHECK( cudaMalloc((void**)&d_outputSDF, numVertices * sizeof(float)) );
    CUDA_CHECK( cudaMemset(d_outputSDF, 0, numVertices * sizeof(float)) );

    Params params = {};
    params.vertices     = d_vertices;
    params.normals      = d_normals;
    params.outputSDF    = d_outputSDF;
    params.localRays    = d_localRays; // Trỏ tới mảng tia Fibonacci
    params.bvhHandle    = bvhHandle;
    params.raysPerPoint = raysPerPoint;
    params.padding      = 0; // Gán rác

    CUdeviceptr d_params;
    CUDA_CHECK( cudaMalloc((void**)&d_params, sizeof(Params)) );
    CUDA_CHECK( cudaMemcpy((void*)d_params, &params, sizeof(Params), cudaMemcpyHostToDevice) );

    std::cout << "OptiX: Đang bắn " << raysPerPoint << " tia Fibonacci/đỉnh cho " << numVertices << " đỉnh...\n";

    OPTIX_CHECK( optixLaunch( pipeline, 0, d_params, sizeof(Params), &sbt, numVertices, 1, 1 ) );
    CUDA_CHECK( cudaDeviceSynchronize() );

    // =========================================================================
    // 5. COPY KẾT QUẢ VỀ CPU VÀ DỌN DẸP
    // =========================================================================
    std::vector<float> results(numVertices);
    CUDA_CHECK( cudaMemcpy(results.data(), d_outputSDF, numVertices * sizeof(float), cudaMemcpyDeviceToHost) );

    cudaFree(d_vertices); cudaFree(d_normals); cudaFree(d_indices);
    cudaFree(d_localRays); // Dọn dẹp mảng tia mẫu
    cudaFree((void*)d_gasOutputBuffer); cudaFree((void*)d_rgSbt);
    cudaFree((void*)d_msSbt); cudaFree((void*)d_hgSbt);
    cudaFree((void*)d_params); cudaFree(d_outputSDF);

    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(raygenProgGroup);
    optixProgramGroupDestroy(missProgGroup);
    optixProgramGroupDestroy(hitProgGroup);
    optixModuleDestroy(module);
    optixDeviceContextDestroy(context);

    return results;
}

// -----------------------------------------------------------------------------
// HÀM CHÍNH GỌI TỪ NGOÀI VÀO
// -----------------------------------------------------------------------------
inline void CaculatingSDFUsingOptix(Model& model, const std::string& shader, int input_raysPerPoint = 64, float input_angle = 150.0) {
    std::cout << "Bắt đầu tính toán SDF bằng OptiX...\n";
    int raysPerPoint = input_raysPerPoint;
    float coneAngle = input_angle * (3.14159265f / 180.0f);

    Matrix& vertices = model.GetVertexMatrix();
    Matrix& indices = model.GetVertexIndicesMatrix();

    auto start = std::chrono::high_resolution_clock::now();

    model.UpdateNormal();
    Matrix& vNormal = model.GetVertexNormalMatrix();

    std::vector<float> sdfResults = RunOptixConeRayCasting(vertices, indices, vNormal, raysPerPoint, coneAngle, shader);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;

    std::cout << "Thời gian chạy OptiX: " << duration.count() << " giây\n";

    double maxDist = 0.0;
    for (int i = 0; i < vertices.Width; ++i) {
        double avgDist = sdfResults[i];
        if (avgDist > maxDist) maxDist = avgDist;
        model.AddHeatMapVertexForPreviewEngine(i, avgDist);
    }

    std::cout << "Hoàn tất SDF! Khoảng cách trung bình lớn nhất: " << maxDist << "\n";
    model.AddToScene("Optix_SDF_Model", false);
}
#endif