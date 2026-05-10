#ifndef CORE_HELPER_H
#define CORE_HELPER_H
#include <string>
#include <fstream>
#include <CL/cl.h>
#include <iostream>
#include <vector>
#define CL_CHECK(call) \
do { \
cl_int err = call; \
if (err != CL_SUCCESS) { \
std::cerr << "Lỗi OpenCL tại dòng " << __LINE__ << ", mã lỗi: " << err << "\n"; \
exit(1); \
} \
} while (0)

inline std::string readFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.good()) throw std::runtime_error("Không thể mở file PTX: " + filepath);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}
// ==============================================================================
struct OpenCLEnvironment {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_device_id device = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel_raycast = nullptr;
    cl_kernel kernel_average = nullptr;
    bool isInitialized = false;
};

inline OpenCLEnvironment InitOpenCLEnvironment(const std::string& kernelSourceCode) {
    std::cout << "[HỆ THỐNG] Đang khởi tạo môi trường OpenCL...\n";


    OpenCLEnvironment env;

    // Tìm Card NVIDIA
    cl_uint num_platforms;
    CL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));
    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), NULL));

    cl_platform_id selected_platform = platforms[0];
    for (int i = 0; i < num_platforms; ++i) {
        char pName[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, pName, NULL);
        if (std::string(pName).find("NVIDIA") != std::string::npos) {
            selected_platform = platforms[i];
            break;
        }
    }

    CL_CHECK(clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 1, &env.device, NULL));
    env.context = clCreateContext(NULL, 1, &env.device, NULL, NULL, NULL);
    env.queue = clCreateCommandQueue(env.context, env.device, 0, NULL);

    // Biên dịch Kernel
    const char* source_str = kernelSourceCode.c_str();
    size_t source_size = kernelSourceCode.length();
    env.program = clCreateProgramWithSource(env.context, 1, &source_str, &source_size, NULL);
    cl_int build_err = clBuildProgram(env.program, 1, &env.device, NULL, NULL, NULL);

    if (build_err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(env.program, env.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(env.program, env.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "LỖI BIÊN DỊCH OPENCL KERNEL:\n" << log.data() << "\n";
        exit(1);
    }

    cl_int err;
    env.kernel_raycast = clCreateKernel(env.program, "sdf_raycast", &err);
    if (err != CL_SUCCESS) { std::cerr << "Lỗi: Không tìm thấy 'sdf_raycast'!\n"; exit(1); }

    env.kernel_average = clCreateKernel(env.program, "sdf_average", &err);
    if (err != CL_SUCCESS) { std::cerr << "Lỗi: Không tìm thấy 'sdf_average'!\n"; exit(1); }

    env.isInitialized = true;

    return env;
}

// Hàm dọn dẹp (Gọi khi tắt chương trình)
inline void ReleaseOpenCLEnvironment(OpenCLEnvironment& env) {
    if (!env.isInitialized) return;
    clReleaseKernel(env.kernel_raycast);
    clReleaseKernel(env.kernel_average);
    clReleaseProgram(env.program);
    clReleaseCommandQueue(env.queue);
    clReleaseContext(env.context);
    env.isInitialized = false;
}

#endif

