//
// Created by tommydatlc on 3/17/26.
//

#ifndef OPTIMIZESDF_MATRIX_H
#define OPTIMIZESDF_MATRIX_H
#define CUBLASAPI
#include <cublas_api.h>
#include <vector>

class Matrix {
private:
    cublasHandle_t cublasHandle;
    float* hostMemory = nullptr;
    float* deviceMemory = nullptr;
    long long totalElement = 0.0f;
    size_t totalSizeInMemory = 0;

    void mul(const Matrix &A, const Matrix &B, float **device_dest);
    float Get(int h,int w,float* mem_Region);
    void Set(int h,int w,float value,float* mem_Region);
    Matrix(float* deviceMemory);
public:
    // Move constructor
    Matrix();
    // ----- REMEMBER TO PUT ANOTHER POINTER HERE
    Matrix(Matrix&& others) {
        this->cublasHandle = others.cublasHandle;
        this->hostMemory = others.hostMemory;
        this->deviceMemory = others.deviceMemory;
        this->totalElement = others.totalElement;
        this->totalSizeInMemory = others.totalSizeInMemory;
        this->Width = others.Width;
        this->Height = others.Height;
        // Important
        others.hostMemory = nullptr;
        others.deviceMemory = nullptr;
    }
    ~Matrix();

    void CopyToHost();

    int Height = 0,Width = 0;
    void CopyToDevice();
    bool IsInit();
    Matrix(int input_height, int input_width, cublasHandle_t cublas_handle_input);
    Matrix& operator*(const Matrix &M);
    void Print();
    void PrintOnGPU();
    float GetHost(int h,int w);
    void SetHost(int h,int w,float value);
    __device__ float GetDevice(int h, int w);
    __device__ void SetDevice(int h, int w, float value);
};
inline std::vector<Matrix*> matrixList;

#endif //OPTIMIZESDF_MATRIX_H