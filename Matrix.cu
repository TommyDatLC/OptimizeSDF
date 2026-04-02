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
    Get(h,w,hostMemory);
}
float Matrix::GetDevice(int h, int w) {
    Get(h,w,deviceMemory);
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


