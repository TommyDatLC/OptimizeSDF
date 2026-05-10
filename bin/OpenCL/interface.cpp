#ifndef OPENCL_SDF_HELPER_H
#define OPENCL_SDF_HELPER_H

#include "../../Core/Helper.hpp"
#include "../../Core/Model.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

// Nhúng bộ xây dựng Octree
#include "octree.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

struct PreprocessedData {
    std::vector<HostFloat4> triangleTexture;
    std::vector<HostFloat4> origins;
    std::vector<HostFloat4> normals;
    std::vector<HostFloat4> tangents;
    std::vector<HostFloat4> binormals;

    std::vector<Octree::GPUNode> octreeNodes;
    std::vector<LeafData> leafDataArray;

    float maxSize;
    int numVertices;
    int numFaces;
};

// ==============================================================================
// 1. MACRO Bắt lỗi an toàn
// (Nếu trong Helper.hpp đã có thì Macro này sẽ bị ghi đè an toàn)
// ==============================================================================
#ifndef CL_CHECK
#define CL_CHECK(call) \
    do { \
        cl_int _cl_err_code = (call); \
        if (_cl_err_code != CL_SUCCESS) { \
            std::cerr << "Lỗi OpenCL tại dòng " << __LINE__ << ", mã lỗi: " << _cl_err_code << "\n"; \
            exit(1); \
        } \
    } while (0)
#endif

// ==============================================================================
// 2. HÀM TIỀN XỬ LÝ (BUILD OCTREE TRÊN CPU)
// ==============================================================================
inline PreprocessedData PreprocessDataForOpenCL(Matrix& vertices, Matrix& indices, Matrix& vNormal) {
    PreprocessedData data;
    data.numVertices = vertices.Width;
    data.numFaces = indices.Width;

    float minX = 1e9, minY = 1e9, minZ = 1e9;
    float maxX = -1e9, maxY = -1e9, maxZ = -1e9;

    data.origins.resize(data.numVertices);
    data.normals.resize(data.numVertices);
    data.tangents.resize(data.numVertices);
    data.binormals.resize(data.numVertices);

    for(int i = 0; i < data.numVertices; i++) {
        float vx = vertices.GetHost(0, i), vy = vertices.GetHost(1, i), vz = vertices.GetHost(2, i);

        minX = std::min(minX, vx); minY = std::min(minY, vy); minZ = std::min(minZ, vz);
        maxX = std::max(maxX, vx); maxY = std::max(maxY, vy); maxZ = std::max(maxZ, vz);

        data.origins[i] = {vx, vy, vz, 0.0f};
        float nx = vNormal.GetHost(0, i), ny = vNormal.GetHost(1, i), nz = vNormal.GetHost(2, i);
        data.normals[i] = {-nx, -ny, -nz, 0.0f};

        HostFloat4 t, b;
        if (abs(nx) > 0.9f) t = {0.0f, 1.0f, 0.0f, 0.0f};
        else t = {1.0f, 0.0f, 0.0f, 0.0f};

        b.x = t.y * data.normals[i].z - t.z * data.normals[i].y;
        b.y = t.z * data.normals[i].x - t.x * data.normals[i].z;
        b.z = t.x * data.normals[i].y - t.y * data.normals[i].x;
        b.w = 0.0f;

        t.x = data.normals[i].y * b.z - data.normals[i].z * b.y;
        t.y = data.normals[i].z * b.x - data.normals[i].x * b.z;
        t.z = data.normals[i].x * b.y - data.normals[i].y * b.x;
        t.w = 0.0f;

        data.tangents[i] = t;
        data.binormals[i] = b;
    }
    data.maxSize = std::max({maxX - minX, maxY - minY, maxZ - minZ});

    std::cout << "    + Đang xây dựng cây Octree...\n";

    std::vector<Octree::Triangle> rawTriangles(data.numFaces);
    for(int i = 0; i < data.numFaces; i++) {
        int i0 = indices.GetHost(0, i), i1 = indices.GetHost(1, i), i2 = indices.GetHost(2, i);
        rawTriangles[i] = {
            {vertices.GetHost(0, i0), vertices.GetHost(1, i0), vertices.GetHost(2, i0)},
            {vertices.GetHost(0, i1), vertices.GetHost(1, i1), vertices.GetHost(2, i1)},
            {vertices.GetHost(0, i2), vertices.GetHost(1, i2), vertices.GetHost(2, i2)}
        };
    }

    Octree::AABB rootBox = {
        {minX - 0.01f, minY - 0.01f, minZ - 0.01f},
        {maxX + 0.01f, maxY + 0.01f, maxZ + 0.01f}
    };

    Octree::Builder builder(rawTriangles, 10, 32);
    auto rootNode = builder.BuildTree(rootBox);

    Octree::FlattenOctree(rootNode.get(), rawTriangles,
                          data.octreeNodes, data.leafDataArray, data.triangleTexture);

    std::cout << "    + Hoàn tất Octree! Tổng số Node: " << data.octreeNodes.size() << "\n";

    return data;
}

// ==============================================================================
// 3. HÀM CHẠY OPENCL CORE
// ==============================================================================
inline std::vector<float> RunOpenCLConeRayCasting(
    const OpenCLEnvironment& env, const PreprocessedData& data,
    int raysPerPoint, float coneAngleRadian)
{
    int totalRays = data.numVertices * raysPerPoint;

    // KIỂM TRA DỮ LIỆU RỖNG: Ngăn OpenCL sụp đổ do cấp phát 0 byte
    if (data.triangleTexture.empty() || data.numVertices == 0) {
        std::cerr << "CẢNH BÁO: Mô hình không có bề mặt (0 faces) để thực hiện SDF Raycasting!\n";
        return std::vector<float>(data.numVertices, 0.0f);
    }

    auto time_prep_start = std::chrono::high_resolution_clock::now();
    cl_int err_code;

    cl_mem d_triangleTexture = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.triangleTexture.size() * sizeof(HostFloat4), (void*)data.triangleTexture.data(), &err_code); CL_CHECK(err_code);
    cl_mem d_octreeNodes = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.octreeNodes.size() * sizeof(Octree::GPUNode), (void*)data.octreeNodes.data(), &err_code); CL_CHECK(err_code);
    cl_mem d_leafDataArray = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.leafDataArray.size() * sizeof(LeafData), (void*)data.leafDataArray.data(), &err_code); CL_CHECK(err_code);
    cl_mem d_origins = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.origins.size() * sizeof(HostFloat4), (void*)data.origins.data(), &err_code); CL_CHECK(err_code);
    cl_mem d_normals = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.normals.size() * sizeof(HostFloat4), (void*)data.normals.data(), &err_code); CL_CHECK(err_code);
    cl_mem d_tangents = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.tangents.size() * sizeof(HostFloat4), (void*)data.tangents.data(), &err_code); CL_CHECK(err_code);
    cl_mem d_binormals = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.binormals.size() * sizeof(HostFloat4), (void*)data.binormals.data(), &err_code); CL_CHECK(err_code);

    cl_mem d_outDistances = clCreateBuffer(env.context, CL_MEM_READ_WRITE, totalRays * sizeof(float), NULL, &err_code); CL_CHECK(err_code);
    cl_mem d_finalSDF = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY, data.numVertices * sizeof(float), NULL, &err_code); CL_CHECK(err_code);

    if (!d_triangleTexture || !d_octreeNodes || !d_leafDataArray || !d_outDistances) {
        std::cerr << "[Lỗi] clCreateBuffer thất bại (trả về NULL)! Không đủ VRAM.\n";
        exit(1);
    }

    cl_uint numRays = raysPerPoint;
    cl_float coneAmp = coneAngleRadian;
    cl_float maxSize = data.maxSize;
    cl_uint u_numVertices = data.numVertices;

    // --- CẤU HÌNH ARGUMENT (CHUẨN XÁC 11 THAM SỐ) ---
    CL_CHECK(clSetKernelArg(env.kernel_raycast, 0, sizeof(cl_mem), &d_triangleTexture));
    CL_CHECK(clSetKernelArg(env.kernel_raycast, 1, sizeof(cl_mem), &d_octreeNodes));
    CL_CHECK(clSetKernelArg(env.kernel_raycast, 2, sizeof(cl_mem), &d_leafDataArray));
    CL_CHECK(clSetKernelArg(env.kernel_raycast, 3, sizeof(cl_mem), &d_origins));
    CL_CHECK(clSetKernelArg(env.kernel_raycast, 4, sizeof(cl_mem), &d_normals));
    CL_CHECK(clSetKernelArg(env.kernel_raycast, 5, sizeof(cl_mem), &d_tangents));
    CL_CHECK(clSetKernelArg(env.kernel_raycast, 6, sizeof(cl_mem), &d_binormals));
    CL_CHECK(clSetKernelArg(env.kernel_raycast, 7, sizeof(cl_mem), &d_outDistances));
    CL_CHECK(clSetKernelArg(env.kernel_raycast, 8, sizeof(cl_uint), &numRays));
    CL_CHECK(clSetKernelArg(env.kernel_raycast, 9, sizeof(cl_float), &coneAmp));
    CL_CHECK(clSetKernelArg(env.kernel_raycast, 10, sizeof(cl_float), &maxSize));

    CL_CHECK(clSetKernelArg(env.kernel_average, 0, sizeof(cl_mem), &d_outDistances));
    CL_CHECK(clSetKernelArg(env.kernel_average, 1, sizeof(cl_mem), &d_finalSDF));
    CL_CHECK(clSetKernelArg(env.kernel_average, 2, sizeof(cl_uint), &numRays));
    CL_CHECK(clSetKernelArg(env.kernel_average, 3, sizeof(cl_uint), &u_numVertices));

    auto time_prep_end = std::chrono::high_resolution_clock::now();
    std::cout << "  -> [Phân tích] Copy RAM->VRAM & Set Args: " << std::chrono::duration<double>(time_prep_end - time_prep_start).count() << " giây\n";

    // --- BƯỚC 5: THỰC THI GPU ---
    size_t global_work_size_ray[1] = { (size_t)totalRays };
    size_t global_work_size_avg[1] = { (size_t)data.numVertices };

    auto time_gpu_start = std::chrono::high_resolution_clock::now();

    // Để local_work_size là NULL để GPU tự tối ưu
    CL_CHECK(clEnqueueNDRangeKernel(env.queue, env.kernel_raycast, 1, NULL, global_work_size_ray, NULL, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(env.queue, env.kernel_average, 1, NULL, global_work_size_avg, NULL, 0, NULL, NULL));
    CL_CHECK(clFinish(env.queue));

    auto time_gpu_end = std::chrono::high_resolution_clock::now();
    std::cout << "  -> [Phân tích] Thời gian thực thi phần cứng GPU: " << std::chrono::duration<double>(time_gpu_end - time_gpu_start).count() << " giây\n";

    // --- BƯỚC 6: COPY KẾT QUẢ ---
    std::vector<float> finalSDF(data.numVertices, 0.0f);
    CL_CHECK(clEnqueueReadBuffer(env.queue, d_finalSDF, CL_TRUE, 0, data.numVertices * sizeof(float), finalSDF.data(), 0, NULL, NULL));

    clReleaseMemObject(d_triangleTexture); clReleaseMemObject(d_octreeNodes);
    clReleaseMemObject(d_leafDataArray); clReleaseMemObject(d_origins);
    clReleaseMemObject(d_normals); clReleaseMemObject(d_tangents);
    clReleaseMemObject(d_binormals); clReleaseMemObject(d_outDistances);
    clReleaseMemObject(d_finalSDF);

    return finalSDF;
}

// ==============================================================================
// 4. HÀM PROFILING GỌI TỪ NGOÀI VÀO
// ==============================================================================
inline void CalculatingSDFUsingOpenCL(const OpenCLEnvironment& env, Model& model, int input_raysPerPoint = 64, float input_angle = 150.0) {
    std::cout << "Bắt đầu tính toán SDF bằng OpenCL...\n";

    int raysPerPoint = input_raysPerPoint;
    float coneAngle = input_angle * (3.14159265f / 180.0f);

    Matrix& vertices = model.GetVertexMatrix();
    Matrix& indices = model.GetVertexIndicesMatrix();

    model.UpdateNormal();
    Matrix& vNormal = model.GetVertexNormalMatrix();
    vNormal.CopyToHost();

    // 1. TIỀN XỬ LÝ (BAO GỒM BUILD CÂY OCTREE)
    std::cout << "[CPU] Đang tiền xử lý dữ liệu và đóng gói mảng...\n";

    auto start = std::chrono::high_resolution_clock::now();
    PreprocessedData preprocessedData = PreprocessDataForOpenCL(vertices, indices, vNormal);
    // 2. BẮT ĐẦU BẤM GIỜ TỔNG THỂ CỦA GPU

    std::vector<float> sdfResults = RunOpenCLConeRayCasting(env, preprocessedData, raysPerPoint, coneAngle);

    // 3. KẾT THÚC BẤM GIỜ
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;

    std::cout << "Thời gian tổng cộng toàn bộ quy trình OpenCL (Nạp + Thực thi + Trả kết quả): " << duration.count() << " giây\n";

    double maxDist = 0.0;
    for (int i = 0; i < vertices.Width; ++i) {
        double avgDist = sdfResults[i];
        if (avgDist > maxDist) maxDist = avgDist;

        // Đã sửa thành hàm theo yêu cầu của bạn: AddHeatMapVertexForPreviewEngine
        model.AddHeatMapVertexForPreviewEngine(i, avgDist);
    }

    std::cout << "Hoàn tất SDF! Khoảng cách trung bình lớn nhất: " << maxDist << "\n";
 //   model.SetShowHeatMap(true); // Bật heatmap bên Model.cu
    model.AddToScene("OpenCL_SDF_Model", false);
}

#endif // OPENCL_SDF_HELPER_H