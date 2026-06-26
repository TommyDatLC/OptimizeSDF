// src/Optix/OptixRunner.cuh
#ifndef OPTIX_RUNNER_CUH
#define OPTIX_RUNNER_CUH

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include "../../Core/Helper.hpp"
#include "../../Core/Matrix.cuh"
#include "OptixHostUtils.cuh"
#include "SDFKernels.cuh"
#include <vector>
#include <string>
#include <chrono>

namespace OptixRunner {

    inline OptixDeviceContext InitContext() {
        CUDA_CHECK( cudaFree(0) );
        OptixDeviceContext context = nullptr;
        OptixDeviceContextOptions options = {}; 
        options.logCallbackFunction = &context_log_cb; 
        options.logCallbackLevel = 0;
        OPTIX_CHECK( optixDeviceContextCreate( 0, &options, &context ) );
        return context;
    }

    inline OptixTraversableHandle BuildBVH(OptixDeviceContext context, CUdeviceptr d_vertex_ptr, CUdeviceptr d_index_ptr, int numVertices, int numFaces, CUdeviceptr& d_tempBuffer, CUdeviceptr& d_gasOutputBuffer, cudaStream_t stream = 0) {
        OptixBuildInput triangleInput = {};
        triangleInput.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangleInput.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.numVertices   = numVertices;
        triangleInput.triangleArray.vertexBuffers = &d_vertex_ptr;
        triangleInput.triangleArray.indexFormat   = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.numIndexTriplets = numFaces;
        triangleInput.triangleArray.indexBuffer   = d_index_ptr;
        uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING };
        triangleInput.triangleArray.flags         = triangleInputFlags;
        triangleInput.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; 
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        
        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accelOptions, &triangleInput, 1, &gasBufferSizes ) );

        // TODO: We should use async allocators, but cudaMalloc is sync. That's fine.
        CUDA_CHECK( cudaMalloc((void**)&d_tempBuffer, gasBufferSizes.tempSizeInBytes) );
        CUDA_CHECK( cudaMalloc((void**)&d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes) );

        OptixTraversableHandle bvhHandle = 0;
        OPTIX_CHECK( optixAccelBuild( context, stream, &accelOptions, &triangleInput, 1, d_tempBuffer, gasBufferSizes.tempSizeInBytes, d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes, &bvhHandle, nullptr, 0 ) );
        return bvhHandle;
    }

    inline OptixPipeline CreatePipeline(OptixDeviceContext context, const std::string& ptxCode, OptixProgramGroup& raygenProgGroup, OptixProgramGroup& missProgGroup, OptixProgramGroup& hitProgGroup, OptixModule& module) {
        OptixModuleCompileOptions moduleCompileOptions = {};
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        pipelineCompileOptions.usesMotionBlur        = false;
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.numPayloadValues      = 1;
        pipelineCompileOptions.numAttributeValues    = 2;
        pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
        OPTIX_CHECK( optixModuleCreate( context, &moduleCompileOptions, &pipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), nullptr, nullptr, &module ) );

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
        return pipeline;
    }

    inline OptixShaderBindingTable BuildSBT(OptixProgramGroup raygenProgGroup, OptixProgramGroup missProgGroup, OptixProgramGroup hitProgGroup, CUdeviceptr& d_rgSbt, CUdeviceptr& d_msSbt, CUdeviceptr& d_hgSbt) {
        RaygenRecord rgSbt; OPTIX_CHECK( optixSbtRecordPackHeader( raygenProgGroup, &rgSbt ) ); CUDA_CHECK( cudaMalloc((void**)&d_rgSbt, sizeof(RaygenRecord)) ); CUDA_CHECK( cudaMemcpy((void*)d_rgSbt, &rgSbt, sizeof(RaygenRecord), cudaMemcpyHostToDevice) );
        MissRecord msSbt; OPTIX_CHECK( optixSbtRecordPackHeader( missProgGroup, &msSbt ) ); CUDA_CHECK( cudaMalloc((void**)&d_msSbt, sizeof(MissRecord)) ); CUDA_CHECK( cudaMemcpy((void*)d_msSbt, &msSbt, sizeof(MissRecord), cudaMemcpyHostToDevice) );
        HitgroupRecord hgSbt; OPTIX_CHECK( optixSbtRecordPackHeader( hitProgGroup, &hgSbt ) ); CUDA_CHECK( cudaMalloc((void**)&d_hgSbt, sizeof(HitgroupRecord)) ); CUDA_CHECK( cudaMemcpy((void*)d_hgSbt, &hgSbt, sizeof(HitgroupRecord), cudaMemcpyHostToDevice) );

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = d_rgSbt; 
        sbt.missRecordBase = d_msSbt; sbt.missRecordStrideInBytes = sizeof( MissRecord ); sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = d_hgSbt; sbt.hitgroupRecordStrideInBytes = sizeof( HitgroupRecord ); sbt.hitgroupRecordCount = 1;
        return sbt;
    }

    inline float host_radicalInverse_VdC(unsigned int bits) {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return float(bits) * 2.3283064365386963e-10f;
    }

    inline float* LaunchOptixAndCUB(
        int numVertices, int raysPerPoint, float coneAngleRadian,
        CUdeviceptr d_vertex_ptr, float3* d_normals,
        OptixTraversableHandle bvhHandle, OptixPipeline pipeline, const OptixShaderBindingTable& sbt) 
    {
        auto t_alloc_optix_start = std::chrono::high_resolution_clock::now();
        float *d_outputDistances, *d_outputWeights;
        int *d_validCounts;
        CUDA_CHECK( cudaMalloc((void**)&d_outputDistances, numVertices * raysPerPoint * sizeof(float)) );
        CUDA_CHECK( cudaMalloc((void**)&d_outputWeights, numVertices * raysPerPoint * sizeof(float)) );
        CUDA_CHECK( cudaMalloc((void**)&d_validCounts, numVertices * sizeof(int)) );

        Params params = {};
        params.vertices     = (float3*)d_vertex_ptr; 
        params.normals      = d_normals;
        params.outputDistances = d_outputDistances;
        params.outputWeights = d_outputWeights;
        params.validCounts  = d_validCounts;
        params.bvhHandle    = bvhHandle;
        params.raysPerPoint = raysPerPoint;
        params.coneAngleRad = coneAngleRadian;

        int limit = (raysPerPoint > 128) ? 128 : raysPerPoint;
        for (int i = 0; i < limit; i++) {
            params.hammersleyUVs[i].x = float(i) / float(limit);
            params.hammersleyUVs[i].y = host_radicalInverse_VdC(i);
        }

        CUdeviceptr d_params;
        CUDA_CHECK( cudaMalloc((void**)&d_params, sizeof(Params)) );
        CUDA_CHECK( cudaMemcpy((void*)d_params, &params, sizeof(Params), cudaMemcpyHostToDevice) );
        auto t_alloc_optix_end = std::chrono::high_resolution_clock::now();
        std::cout << "[DEBUG] [SDF Pipeline] Thời gian Cấp phát bộ nhớ cho OptiX (cudaMalloc + Memcpy): " << std::chrono::duration<double>(t_alloc_optix_end - t_alloc_optix_start).count() << " giây\n";

        std::cout << "OptiX: Đang bắn " << raysPerPoint << " tia Hammersley Uniform/đỉnh...\n";
        auto t_start = std::chrono::high_resolution_clock::now();
        OPTIX_CHECK( optixLaunch( pipeline, 0, d_params, sizeof(Params), &sbt, numVertices, 1, 1 ) );
        CUDA_CHECK( cudaDeviceSynchronize() );
        auto t_optix_done = std::chrono::high_resolution_clock::now();
        std::cout << "[DEBUG] [SDF Pipeline] Thời gian OptiX Ray Tracing thực tế: " << std::chrono::duration<double>(t_optix_done - t_start).count() << " giây\n";

        // ---------------------------------------------------------------------
        // THAY VÌ CUB SORT CHẬM CHẠP, SẮP XẾP TRỰC TIẾP TRONG KERNEL TÍNH SDF
        // ---------------------------------------------------------------------
        auto t_raw_sdf_start = std::chrono::high_resolution_clock::now();
        float* d_rawSDF;
        CUDA_CHECK( cudaMalloc((void**)&d_rawSDF, numVertices * sizeof(float)) );
        
        int blockSize = 256;
        int gridSizeSDF = (numVertices + blockSize - 1) / blockSize;
        GPUComputeRawSDF<<<gridSizeSDF, blockSize>>>(
            d_outputDistances, d_outputWeights, d_validCounts, d_rawSDF, numVertices, raysPerPoint
        );
        CUDA_CHECK( cudaDeviceSynchronize() );
        auto t_raw_sdf_end = std::chrono::high_resolution_clock::now();
        std::cout << "[DEBUG] [SDF Pipeline] Thời gian tính toán SDF thô (Bao gồm Thread-local Sort): " << std::chrono::duration<double>(t_raw_sdf_end - t_raw_sdf_start).count() << " giây\n";

        cudaFree((void*)d_params);
        cudaFree((void*)d_outputDistances); cudaFree((void*)d_outputWeights); cudaFree((void*)d_validCounts);

        return d_rawSDF;
    }
}
#endif // OPTIX_RUNNER_CUH
