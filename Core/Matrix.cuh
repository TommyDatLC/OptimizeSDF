#ifndef OPTIMIZESDF_MATRIX_H
#define OPTIMIZESDF_MATRIX_H
#include <cublas_api.h>
#include <vector>

template <typename T>
class Matrix {
    friend class MatrixMemoryManager;
private:
    bool isOwned = true;
    cublasHandle_t cublasHandle;
    T* hostMemory = nullptr;
    T* deviceMemory = nullptr;
    long long totalElement = 0.0f;
    size_t totalSizeInMemory = 0;

    void mul(const Matrix<T> &A, const Matrix<T> &B, T **device_dest);
    T Get(int h,int w,T* mem_Region);
    void Set(int h,int w,T value,T* mem_Region);
    Matrix(T* deviceMemory);
public:
    // Move constructor
    Matrix() {}
    ~Matrix();

    int Height = 0,Width = 0;
    void CopyToHost();
    void CopyToDevice();
    bool IsInit();
    Matrix(int input_height, int input_width, cublasHandle_t cublas_handle_input);
    Matrix<T>& operator*(const Matrix<T> &M);
    void Print();
    void PrintOnGPU();
    T GetHost(int h,int w);
    void SetHost(int h,int w, T value);
    T* getDevicePtr() { return deviceMemory; }
    T* getHostPtr() { return hostMemory; }
    void CPUInverse();
    void CPUTranspose();
    size_t GetSize();
};

#endif //OPTIMIZESDF_MATRIX_H