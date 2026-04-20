//
// Created by tommydatlc on 3/17/26.
//

#define CUBLAS_API
#include "Matrix.cuh"
#include <cstdio>
#include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include "MatrixUtils.cuh"
#include "glm/gtc/type_ptr.hpp"

Matrix::Matrix(const int input_height,
               const int input_width,
               cublasHandle_t cublas_handle_input) {
    totalElement = input_height*input_width;
    hostMemory = new float[totalElement] {0};
    totalSizeInMemory = sizeof(float) * totalElement;
    Width = input_width;
    Height = input_height;
    cudaMalloc((void**)&deviceMemory,totalSizeInMemory);
    cublasHandle = cublas_handle_input;
}

void Matrix::mul(const Matrix &A,const Matrix &B, float **device_dest) {
    float alpha = 1;
    float beta = 0;
    if (A.Width != B.Height) {
        std::cout << A.Width << " != " << A.Height << '\n';
        std::__throw_runtime_error("Dimensions A.Width != B.Height");
    }
    int m = A.Height;
    int k = A.Width;
    int n = B.Width;
    cudaMalloc((void**)device_dest,sizeof(float) * m * n);
    if (A.deviceMemory == nullptr || B.deviceMemory == nullptr) {
        std::__throw_runtime_error("device memory has been not allocated");
    }
    cublasSgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N,
        m,n,k,&alpha,
        deviceMemory,Height,
        B.deviceMemory,B.Height,&beta,
       // C.deviceMemory,M.Width
       *device_dest,m
       );
    cudaDeviceSynchronize();
}

Matrix::~Matrix() {
    if (!isOwned)
        return;
    cudaFree(deviceMemory);
    delete hostMemory;
}
void Matrix::CopyToHost() {
    cudaMemcpy(hostMemory,deviceMemory,totalSizeInMemory,cudaMemcpyDeviceToHost);
}
void Matrix::CopyToDevice() {
    cudaMemcpy(deviceMemory,hostMemory,totalSizeInMemory,cudaMemcpyHostToDevice);
}

bool Matrix::IsInit() {
    return false;
}

Matrix& Matrix::operator*(const Matrix &M) {
    float* deviceResultmemory = nullptr;
    mul(*this,M,&deviceResultmemory);
    Matrix *result = new Matrix(Height,M.Width,cublasHandle);
    result->deviceMemory = deviceResultmemory;
    Matrix& resRefer = *result;
    matrixList.push_back(result);
    return resRefer;
}

void Matrix::Print() {
    printMemory(hostMemory,Height,Width);
}

void Matrix::PrintOnGPU() {
    printOnGPUKernel<<<1,1>>>(deviceMemory,Height,Width);
    cudaDeviceSynchronize();
}

float Matrix::GetHost(int h, int w) {
    return Get(h,w,hostMemory);
}
float Matrix::GetDevice(int h, int w) {
    return Get(h,w,deviceMemory);
}
float* Matrix::getDevicePtr() const { return deviceMemory; }

void Matrix::CPUTranspose() {
    // 1. Đồng bộ dữ liệu từ GPU về CPU trước khi xử lý
    if (Height == 4 && Width == 4) {
        // Tối ưu hóa cho ma trận 4x4 (thường dùng cho View/Model Matrix)
        glm::mat4 m = glm::make_mat4(hostMemory);
        m = glm::transpose(m);
        std::memcpy(hostMemory, glm::value_ptr(m), totalSizeInMemory);
    } else if (Height == 3 && Width == 3) {
        glm::mat3 m = glm::make_mat3(hostMemory);
        m = glm::transpose(m);
        std::memcpy(hostMemory, glm::value_ptr(m), totalSizeInMemory);
    }
    // 2. Đẩy dữ liệu đã xử lý ngược lại GPU

}
void Matrix::CPUInverse() {
    if (Height != Width) {
        throw std::runtime_error("Phep nghich dao chi ap dung cho ma tran vuong.");
    }
    if (Height == 4) {
        glm::mat4 m = glm::make_mat4(hostMemory);
        m = glm::inverse(m);
        std::memcpy(hostMemory, glm::value_ptr(m), totalSizeInMemory);
    } else if (Height == 3) {
        glm::mat3 m = glm::make_mat3(hostMemory);
        m = glm::inverse(m);
        std::memcpy(hostMemory, glm::value_ptr(m), totalSizeInMemory);
    } else {
        throw std::runtime_error("Hien tai GLM chi ho tro nghich dao mat3 va mat4 trong code nay.");
    }
}
size_t Matrix::GetSize() {
    return totalSizeInMemory;
}

void Matrix::SetHost(int h, int w, float value) {
    Set(h,w,value,hostMemory);
}

void Matrix::SetDevice(int h, int w, float value) {
    Set(h,w,value,deviceMemory);
}
float Matrix::Get(int h, int w,float* mem_Region) {
    return mem_Region[w * Height + h];
}

void Matrix::Set(int h, int w,float value,float* mem_Region) {
    if (h >= Height || w >= Width) {
        std::__throw_runtime_error("Height or Width out of range");
    }
    mem_Region[w * Height + h] = value;
}


