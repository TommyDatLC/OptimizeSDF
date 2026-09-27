#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "../../Core/Matrix.cuh"
#include "../../Core/Helper.hpp"
#include "../../Core/ModelHelper.cu"
#include "OptixHostUtils.cuh"
#include "SDFKernels.cuh"
#include "OptixRunner.cuh"

struct PyOptixState {
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
    bool initialized = false;
};

static PyOptixState g_pyOptixState;

extern "C" __declspec(dllexport) int ComputeSDF_API(
    const float* h_vertices,
    int num_vertices,
    const unsigned int* h_faces,
    int num_faces,
    bool use_post_processing,
    int rays_per_point,
    float cone_angle_deg,
    const char* ptx_path,
    float* h_out_sdf
) {
    if (!h_vertices || num_vertices <= 0 || !h_faces || num_faces <= 0 || !h_out_sdf) {
        std::cerr << "[ComputeSDF_API] Invalid input parameters.\n";
        return -1;
    }

    try {
        if (!g_pyOptixState.initialized) {
            optixInit();
            std::string ptxFile = (ptx_path && ptx_path[0] != '\0') ? std::string(ptx_path) : "SDFOptix.ptx";
            std::string ptxCode = readFile(ptxFile);
            if (ptxCode.empty()) {
                std::cerr << "[ComputeSDF_API] Failed to read PTX file: " << ptxFile << "\n";
                return -3;
            }
            g_pyOptixState.context = OptixRunner::InitContext();
            g_pyOptixState.module = nullptr;
            g_pyOptixState.pipeline = OptixRunner::CreatePipeline(
                g_pyOptixState.context, ptxCode,
                g_pyOptixState.raygenProgGroup, g_pyOptixState.missProgGroup,
                g_pyOptixState.hitProgGroup, g_pyOptixState.module
            );
            g_pyOptixState.sbt = OptixRunner::BuildSBT(
                g_pyOptixState.raygenProgGroup, g_pyOptixState.missProgGroup,
                g_pyOptixState.hitProgGroup, g_pyOptixState.d_rgSbt,
                g_pyOptixState.d_msSbt, g_pyOptixState.d_hgSbt
            );
            g_pyOptixState.initialized = true;
        }

        Matrix<float> vMat(num_vertices, 3, nullptr);
        for (int i = 0; i < num_vertices; ++i) {
            vMat.SetHost(i, 0, h_vertices[i * 3 + 0]);
            vMat.SetHost(i, 1, h_vertices[i * 3 + 1]);
            vMat.SetHost(i, 2, h_vertices[i * 3 + 2]);
        }
        vMat.CopyToDevice();

        Matrix<unsigned int> iMat(num_faces, 3, nullptr);
        for (int i = 0; i < num_faces; ++i) {
            iMat.SetHost(i, 0, h_faces[i * 3 + 0]);
            iMat.SetHost(i, 1, h_faces[i * 3 + 1]);
            iMat.SetHost(i, 2, h_faces[i * 3 + 2]);
        }
        iMat.CopyToDevice();

        Matrix<float> nMat(num_vertices, 3, nullptr);

        cudaStream_t streamNormal, streamBVH;
        CUDA_CHECK(cudaStreamCreate(&streamNormal));
        CUDA_CHECK(cudaStreamCreate(&streamBVH));

        CUDA_CHECK(cudaMemsetAsync(nMat.getDevicePtr(), 0, nMat.GetSize(), streamNormal));

        int blockSize = 256;
        int gridSizeFaces = (num_faces + blockSize - 1) / blockSize;
        GPUNormalCaculation<<<gridSizeFaces, blockSize, 0, streamNormal>>>(
            (const float3*)vMat.getDevicePtr(),
            (const uint3*)iMat.getDevicePtr(),
            num_faces,
            (float3*)nMat.getDevicePtr()
        );

        int gridSizeVerts = (num_vertices + blockSize - 1) / blockSize;
        GPUNormalizeVertexNormal<<<gridSizeVerts, blockSize, 0, streamNormal>>>(
            (float3*)nMat.getDevicePtr(),
            num_vertices
        );

        float coneAngleRadian = cone_angle_deg * (3.14159265f / 180.0f);
        CUdeviceptr d_tempBuffer, d_gasOutputBuffer;
        OptixTraversableHandle bvhHandle = OptixRunner::BuildBVH(
            g_pyOptixState.context,
            (CUdeviceptr)vMat.getDevicePtr(),
            (CUdeviceptr)iMat.getDevicePtr(),
            num_vertices, num_faces,
            d_tempBuffer, d_gasOutputBuffer, streamBVH
        );

        CUDA_CHECK(cudaStreamSynchronize(streamBVH));
        CUDA_CHECK(cudaStreamSynchronize(streamNormal));

        float* d_rawSDF = OptixRunner::LaunchOptixAndCUB(
            num_vertices, rays_per_point, coneAngleRadian,
            (CUdeviceptr)vMat.getDevicePtr(), (float3*)nMat.getDevicePtr(),
            bvhHandle, g_pyOptixState.pipeline, g_pyOptixState.sbt
        );

        CUDA_CHECK(cudaFree((void*)d_tempBuffer));
        CUDA_CHECK(cudaFree((void*)d_gasOutputBuffer));
        CUDA_CHECK(cudaStreamDestroy(streamNormal));
        CUDA_CHECK(cudaStreamDestroy(streamBVH));

        if (!use_post_processing) {
            CUDA_CHECK(cudaMemcpy(h_out_sdf, d_rawSDF, num_vertices * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_rawSDF));
            return 0;
        }

        // Post-Processing: CSR Graph building and Anisotropic Smoothing
        cudaStream_t streamCSR, streamNorm;
        CUDA_CHECK(cudaStreamCreate(&streamCSR));
        CUDA_CHECK(cudaStreamCreate(&streamNorm));

        int numEdges = num_faces * 6;
        uint64_t *d_edges, *d_sortedEdges, *d_uniqueEdges;
        int *d_numUniqueEdges, *d_nbrOffsets, *d_nbrLists;
        CUDA_CHECK(cudaMalloc(&d_edges, numEdges * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_sortedEdges, numEdges * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_uniqueEdges, numEdges * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_numUniqueEdges, sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_nbrOffsets, (num_vertices + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_nbrLists, numEdges * sizeof(int)));

        float* d_minMaxBox;
        CUDA_CHECK(cudaMalloc((void**)&d_minMaxBox, 6 * sizeof(float)));
        float initBox[6] = {1e15f, -1e15f, 1e15f, -1e15f, 1e15f, -1e15f};
        CUDA_CHECK(cudaMemcpyAsync(d_minMaxBox, initBox, 6 * sizeof(float), cudaMemcpyHostToDevice, streamNorm));

        float* d_sdfBuf1 = d_rawSDF;
        float* d_sdfBuf2;
        CUDA_CHECK(cudaMalloc((void**)&d_sdfBuf2, num_vertices * sizeof(float)));

        float* d_minMaxSDF;
        CUDA_CHECK(cudaMalloc((void**)&d_minMaxSDF, 2 * sizeof(float)));
        float initSDF[2] = {1e15f, -1e15f};
        CUDA_CHECK(cudaMemcpyAsync(d_minMaxSDF, initSDF, 2 * sizeof(float), cudaMemcpyHostToDevice, streamNorm));

        GPUGenerateEdges<<<gridSizeFaces, blockSize, 0, streamCSR>>>((const uint3*)iMat.getDevicePtr(), num_faces, d_edges);

        void *d_temp_sort = nullptr; size_t temp_sort_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_temp_sort, temp_sort_bytes, d_edges, d_sortedEdges, numEdges, 0, sizeof(uint64_t)*8, streamCSR);
        CUDA_CHECK(cudaMalloc(&d_temp_sort, temp_sort_bytes));
        cub::DeviceRadixSort::SortKeys(d_temp_sort, temp_sort_bytes, d_edges, d_sortedEdges, numEdges, 0, sizeof(uint64_t)*8, streamCSR);

        void *d_temp_unique = nullptr; size_t temp_unique_bytes = 0;
        cub::DeviceSelect::Unique(d_temp_unique, temp_unique_bytes, d_sortedEdges, d_uniqueEdges, d_numUniqueEdges, numEdges, streamCSR);
        CUDA_CHECK(cudaMalloc(&d_temp_unique, temp_unique_bytes));
        cub::DeviceSelect::Unique(d_temp_unique, temp_unique_bytes, d_sortedEdges, d_uniqueEdges, d_numUniqueEdges, numEdges, streamCSR);

        int numUniqueEdges = 0;
        CUDA_CHECK(cudaMemcpyAsync(&numUniqueEdges, d_numUniqueEdges, sizeof(int), cudaMemcpyDeviceToHost, streamCSR));

        GPUComputeBoundingBox<<<gridSizeVerts, blockSize, 0, streamNorm>>>((const float3*)vMat.getDevicePtr(), num_vertices, d_minMaxBox);
        float h_minMaxBox[6];
        CUDA_CHECK(cudaMemcpyAsync(h_minMaxBox, d_minMaxBox, 6 * sizeof(float), cudaMemcpyDeviceToHost, streamNorm));

        GPUComputeSDFMinMax<<<gridSizeVerts, blockSize, 0, streamNorm>>>(d_sdfBuf1, num_vertices, d_minMaxSDF);
        GPUApplySDFNormalization<<<gridSizeVerts, blockSize, 0, streamNorm>>>(d_sdfBuf1, num_vertices, d_minMaxSDF);

        CUDA_CHECK(cudaStreamSynchronize(streamCSR));
        CUDA_CHECK(cudaStreamSynchronize(streamNorm));

        int gridSizeUnique = (numUniqueEdges + blockSize - 1) / blockSize;
        GPUExtractCSR<<<gridSizeUnique, blockSize, 0, streamCSR>>>(d_uniqueEdges, numUniqueEdges, d_nbrOffsets, d_nbrLists, num_vertices);
        CUDA_CHECK(cudaStreamSynchronize(streamCSR));

        float dx = h_minMaxBox[1] - h_minMaxBox[0];
        float dy = h_minMaxBox[3] - h_minMaxBox[2];
        float dz = h_minMaxBox[5] - h_minMaxBox[4];
        float bboxDiagonal = std::sqrt(dx*dx + dy*dy + dz*dz);

        int numIterations = 3;
        float sigmaSpatial = bboxDiagonal * 0.02f;
        float sigmaRange = 0.1f;
        float3* d_vertices_direct = (float3*)vMat.getDevicePtr();

        for (int iter = 0; iter < numIterations; iter++) {
            float* d_in = (iter % 2 == 0) ? d_sdfBuf1 : d_sdfBuf2;
            float* d_out = (iter % 2 == 0) ? d_sdfBuf2 : d_sdfBuf1;
            AnisotropicSmoothingKernel<<<gridSizeVerts, blockSize>>>(
                d_vertices_direct, d_in, d_out, d_nbrOffsets, d_nbrLists,
                num_vertices, sigmaSpatial, sigmaRange
            );
            cudaDeviceSynchronize();
        }

        float* d_finalOut = (numIterations % 2 == 0) ? d_sdfBuf1 : d_sdfBuf2;
        CUDA_CHECK(cudaMemcpy(h_out_sdf, d_finalOut, num_vertices * sizeof(float), cudaMemcpyDeviceToHost));

        cudaFree(d_edges); cudaFree(d_sortedEdges); cudaFree(d_temp_sort);
        cudaFree(d_uniqueEdges); cudaFree(d_numUniqueEdges); cudaFree(d_temp_unique);
        cudaFree(d_minMaxBox); cudaFree(d_minMaxSDF);
        cudaFree(d_sdfBuf1); cudaFree(d_sdfBuf2);
        cudaFree(d_nbrOffsets); cudaFree(d_nbrLists);
        CUDA_CHECK(cudaStreamDestroy(streamCSR));
        CUDA_CHECK(cudaStreamDestroy(streamNorm));

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ComputeSDF_API Exception]: " << e.what() << std::endl;
        return -2;
    }
}
