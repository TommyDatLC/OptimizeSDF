//
// Created by tommydatlc on 3/17/26.
//
#ifndef CUBLASAPI
#define CUBLASAPI
#endif

#ifndef OPTIMIZESDF_MATRIXMEMORYMANAGER_H
#define OPTIMIZESDF_MATRIXMEMORYMANAGER_H

#include <cublas_api.h>
#include <cublas_v2.h>
#include <memory>
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
#include "Matrix.cuh"

class MatrixMemoryManager {
private:
    // 1. KỸ THUẬT TYPE ERASURE: Lớp cơ sở trừu tượng để chứa mọi loại Template
    struct MatrixWrapper {
        virtual ~MatrixWrapper() = default;
        virtual void* getPtr() = 0;
    };

    // 2. Lớp kế thừa giữ kiểu T thực sự của Ma trận
    template<typename T>
    struct WrapperImpl : public MatrixWrapper {
        Matrix<T>* mat;
        WrapperImpl(Matrix<T>* m) : mat(m) {}

        // Khi Wrapper bị hủy, nó sẽ gọi đúng Destructor của Matrix<T>
        ~WrapperImpl() override {
            delete mat;
        }
        void* getPtr() override { return mat; }
    };

    // Danh sách lưu trữ đa hình
    std::vector<std::unique_ptr<MatrixWrapper>> matrixList;

public:
    cublasHandle_t handle = nullptr;

    MatrixMemoryManager() {
        cublasCreate(&handle);
    }

    ~MatrixMemoryManager() {
        removeAllMatrix();
        if (handle) cublasDestroy(handle);
    }

    // =========================================================================
    // CÁC HÀM TẠO MA TRẬN (ĐÃ CHUYỂN THÀNH TEMPLATE)
    // =========================================================================
    template<typename T>
    Matrix<T>& CreateMatrix(int h, int w) {
        return *CreateMatrixPointer<T>(h, w);
    }

    template<typename T>
    Matrix<T>* CreateMatrixPointer(int h, int w) {
        Matrix<T>* createResult = new Matrix<T>(h, w, handle);
        // Bọc Ma trận vào lớp vỏ và nhét vào danh sách quản lý chung
        matrixList.push_back(std::make_unique<WrapperImpl<T>>(createResult));
        return createResult;
    }

    template<typename T>
    Matrix<T>* CreateMatrixPointer(Matrix<T>& source, bool isDeepCopy) {
        Matrix<T>* createResult = new Matrix<T>();
        createResult->cublasHandle = handle;

        if (isDeepCopy) {
            DeepCopy<T>(*createResult, source);
        } else {
            ShallowCopy<T>(*createResult, source);
        }

        matrixList.push_back(std::make_unique<WrapperImpl<T>>(createResult));
        return createResult;
    }

    // =========================================================================
    // CÁC HÀM COPY (ĐÃ CHUYỂN THÀNH TEMPLATE)
    // =========================================================================
    template<typename T>
    void DeepCopy(Matrix<T>& destination, const Matrix<T>& source) {
        if (destination.deviceMemory != nullptr) cudaFree(destination.deviceMemory);
        if (destination.hostMemory != nullptr) delete[] destination.hostMemory;

        destination.Width = source.Width;
        destination.Height = source.Height;
        destination.totalElement = source.totalElement;
        destination.totalSizeInMemory = source.totalSizeInMemory;
        destination.cublasHandle = source.cublasHandle;

        if (source.hostMemory != nullptr) {
            destination.hostMemory = new T[destination.totalElement];
            std::memcpy(destination.hostMemory, source.hostMemory, destination.totalSizeInMemory);
        } else {
            destination.hostMemory = nullptr;
        }

        if (source.deviceMemory != nullptr) {
            cudaMalloc((void**)&destination.deviceMemory, destination.totalSizeInMemory);
            cudaMemcpy(destination.deviceMemory, source.deviceMemory, destination.totalSizeInMemory, cudaMemcpyDeviceToDevice);
        } else {
            destination.deviceMemory = nullptr;
        }
    }

    template<typename T>
    void ShallowCopy(Matrix<T>& destination, Matrix<T>& source) {
        if (destination.deviceMemory != nullptr) cudaFree(destination.deviceMemory);
        if (destination.hostMemory != nullptr) delete[] destination.hostMemory;

        destination.cublasHandle = source.cublasHandle;
        destination.hostMemory = source.hostMemory;
        destination.deviceMemory = source.deviceMemory;
        destination.totalElement = source.totalElement;
        destination.totalSizeInMemory = source.totalSizeInMemory;
        destination.Width = source.Width;
        destination.Height = source.Height;

        source.hostMemory = nullptr;
        source.deviceMemory = nullptr;
        source.isOwned = false;
        source.Width = 0;
        source.Height = 0;
        source.totalElement = 0;
        source.totalSizeInMemory = 0;
    }

    // =========================================================================
    // HÀM QUẢN LÝ VÀ DỌN DẸP
    // =========================================================================
    void removeAllMatrix() {
        // Hàm clear() của vector sẽ tự động kích hoạt Destructor của std::unique_ptr,
        // từ đó kích hoạt ~WrapperImpl() và gọi đúng lệnh delete Matrix<T> tương ứng!
        matrixList.clear();
    }

    template<typename T>
    void FreeMatrix(Matrix<T>& mat) {
        void* targetPtr = &mat;
        // Quét tìm trong danh sách đa hình
        auto it = std::remove_if(matrixList.begin(), matrixList.end(),
            [targetPtr](const std::unique_ptr<MatrixWrapper>& wrapper) {
                return wrapper->getPtr() == targetPtr;
            });

        if (it != matrixList.end()) {
            matrixList.erase(it, matrixList.end());
        }
    }
};

inline auto matrixMemMang = MatrixMemoryManager();

#endif //OPTIMIZESDF_MATRIXMEMORYMANAGER_H