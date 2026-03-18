//
// Created by tommydatlc on 3/17/26.
//

#ifndef OPTIMIZESDF_MATRIX_H
#define OPTIMIZESDF_MATRIX_H
#include <cublas_api.h>


class Matrix {
private:
    cublasHandle_t cublasHandle;
    float* hostMemory = nullptr;
    float* deviceMemory = nullptr;
    long long totalElement = 0.0f;
    size_t totalSizeInMemory = 0;

    void mul(const Matrix &A, const Matrix &B, float **device_dest);
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
    int Height = 0,Width = 0;
    void CopyToDevice();

    Matrix(int input_height, int input_width, cublasHandle_t cublas_handle_input);
    Matrix operator*(const Matrix &M);
    void Print();
    void PrintOnGPU();
    float Get(int h,int w);
    void Set(int h,int w,float value);
};


#endif //OPTIMIZESDF_MATRIX_H