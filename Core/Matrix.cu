// ... existing code ...
#include <cuda.h>
#include "MatrixUtils.cuh"
#include "glm/gtc/type_ptr.hpp"
#include <type_traits> // Bắt buộc cho if constexpr để rẽ nhánh kiểu T

template <typename T>
Matrix<T>::Matrix(const int input_height,
               const int input_width,
               cublasHandle_t cublas_handle_input) {
    totalElement = input_height*input_width;
    hostMemory = new T[totalElement] {0};
    totalSizeInMemory = sizeof(T) * totalElement;
    Width = input_width;
    Height = input_height;
    cudaMalloc((void**)&deviceMemory,totalSizeInMemory);
    cublasHandle = cublas_handle_input;
}

template <typename T>
Matrix<T>::~Matrix() {
    if (isOwned) {
        if (hostMemory) { delete[] hostMemory; hostMemory = nullptr; }
        if (deviceMemory) { cudaFree(deviceMemory); deviceMemory = nullptr; }
    }
}

template <typename T>
void Matrix<T>::mul(const Matrix<T> &A,const Matrix<T> &B, T **device_dest) {
    T alpha = 1;
    T beta = 0;
    if (A.Width != B.Height) {
        std::cout << A.Width << " != " << A.Height << '\n';
        throw std::runtime_error("Dimensions A.Width != B.Height");
    }
    int m = A.Height;
    int k = A.Width;
    int n = B.Width;

    // Tính năng siêu việt của C++17: Phân nhánh theo kiểu T ở thì Compile-time
    if constexpr (std::is_same_v<T, float>) {
        cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A.deviceMemory, m, B.deviceMemory, k, &beta, *device_dest, m);
    } else if constexpr (std::is_same_v<T, double>) {
        cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A.deviceMemory, m, B.deviceMemory, k, &beta, *device_dest, m);
    } else {
        throw std::runtime_error("Chua ho tro kieu du lieu nay cho cuBLAS");
    }
}
// ... existing code ...
template <typename T>
void Matrix<T>::CPUInverse() {
    if (Height != Width) {
        throw std::runtime_error("Phep nghich dao chi ap dung cho ma tran vuong.");
    }
    if constexpr (std::is_same_v<T, float>) {
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
    // Nếu bạn muốn hỗ trợ double cho GLM, hãy thêm else if constexpr ở đây với glm::dmat4
}

template <typename T>
size_t Matrix<T>::GetSize()
{
    return totalSizeInMemory;
}

template <typename T>
void Matrix<T>::CopyToHost() {
    cudaMemcpy(hostMemory,deviceMemory,totalSizeInMemory,cudaMemcpyDeviceToHost);
}
template <typename T>
void Matrix<T>::CopyToDevice() {
    cudaMemcpy(deviceMemory,hostMemory,totalSizeInMemory,cudaMemcpyHostToDevice);
}
// LƯU Ý: Bạn cần lặp lại cấu trúc template <typename T> phía trên cho TẤT CẢ các hàm còn lại
// (Print, CopyToHost, SetHost, GetHost...) trong file Matrix.cu này.
// ... existing code ...
template <typename T>
void Matrix<T>::SetHost(int h, int w, T value) {
    Set(h,w,value,hostMemory);
}
template <typename T>
void Matrix<T>::SetDevice(int h, int w,T value) {
    Set(h,w,value,deviceMemory);
}
template <typename T>
T Matrix<T>::Get(int h, int w, T* mem_Region) {
    return mem_Region[h * Width + w]; // Đã sửa thành Row-Major
}
template <typename T>
void Matrix<T>::Set(int h, int w,T value,T* mem_Region) {
    if (h >= Height || w >= Width) {
        throw std::runtime_error("Height or Width out of range");
    }
    mem_Region[h * Width + w] = value; // Đã sửa thành Row-Major
}
template <typename T>
T Matrix<T>::GetHost(int h, int w) {
    return Get(h,w,hostMemory);
}
template <typename T>
T Matrix<T>::GetDevice(int h, int w) {
    return Get(h,w,deviceMemory);
}

// DÒNG QUAN TRỌNG NHẤT: Bắt buộc đặt ở cuối cùng của file .cu
// Ép trình biên dịch Instantiate các phiên bản này để C++ Linker có thể nhìn thấy
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<uint32_t>;