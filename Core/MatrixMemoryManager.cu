//
// Created by tommydatlc on 3/17/26.
//

#include "MatrixMemoryManager.cuh"
#include <cublas_v2.h>
#include <memory>
#include <cstring>
#include <iostream>

#include "Matrix.cuh"
// full copy
// Đổi tên tham số thành dest (đích) và src (nguồn) để dễ hiểu
void MatrixMemoryManager::DeepCopy(Matrix &destination, const Matrix &source) {
    // 0. BẮT BUỘC: Dọn dẹp bộ nhớ cũ của destination trước khi copy để tránh Memory Leak
    if (destination.deviceMemory != nullptr) {
        cudaFree(destination.deviceMemory);
    }
    if (destination.hostMemory != nullptr) {
        delete[] destination.hostMemory;
    }

    // 1. Copy các biến giá trị (Value types)
    destination.Width = source.Width;
    destination.Height = source.Height;
    destination.totalElement = source.totalElement;
    destination.totalSizeInMemory = source.totalSizeInMemory;
    destination.cublasHandle = source.cublasHandle;

    // 2. Cấp phát và Copy Host Memory (RAM)
    if (source.hostMemory != nullptr) {
        destination.hostMemory = new float[destination.totalElement];
        std::memcpy(destination.hostMemory, source.hostMemory, destination.totalSizeInMemory);
    } else {
        destination.hostMemory = nullptr;
    }

    // 3. Cấp phát và Copy Device Memory (GPU VRAM)
    if (source.deviceMemory != nullptr) {
        cudaMalloc((void**)&destination.deviceMemory, destination.totalSizeInMemory);
        // Dùng cudaMemcpyDeviceToDevice để copy trực tiếp trong GPU, cực kỳ nhanh
        cudaMemcpy(destination.deviceMemory, source.deviceMemory, destination.totalSizeInMemory, cudaMemcpyDeviceToDevice);
    } else {
        destination.deviceMemory = nullptr;
    }
}

// Phép gán Shallow Copy (Bản chất code của bạn là Move Semantics)
// Bỏ từ khóa 'const' ở tham số source vì chúng ta cần sửa nó thành nullptr
void MatrixMemoryManager::ShallowCopy(Matrix &destination, Matrix &source) {

    std::cerr << "Canh bao: Shallow Copy matrix source van chua bi chuyen khoi matrixList" << std::endl;
    // 0. Dọn dẹp bộ nhớ cũ của destination trước khi "cướp" vùng nhớ của source

    if (destination.deviceMemory != nullptr) {
        cudaFree(destination.deviceMemory);
    }
    if (destination.hostMemory != nullptr) {
        delete[] destination.hostMemory;
    }
    if (destination.deviceMemory == source.deviceMemory) {
        std::cout << "copy 2 ma tran giong nhau";
        return;
    }
    // 1. Gán thẳng các con trỏ và dữ liệu từ source sang destination
    destination.cublasHandle = source.cublasHandle;
    destination.hostMemory = source.hostMemory;
    destination.deviceMemory = source.deviceMemory;
    destination.totalElement = source.totalElement;
    destination.totalSizeInMemory = source.totalSizeInMemory;
    destination.Width = source.Width;
    destination.Height = source.Height;

    // 2. QUAN TRỌNG: Ngắt kết nối từ source để khi source bị hủy, nó không gọi cudaFree
    source.hostMemory = nullptr;
    source.deviceMemory = nullptr;
    source.isOwned = false;
    source.Width = 0;
    source.Height = 0;
    source.totalElement = 0;
    source.totalSizeInMemory = 0;
}
MatrixMemoryManager::MatrixMemoryManager()  {
    cublasCreate(&handle);
}

Matrix &MatrixMemoryManager::CreateMatrix(const int hieght, const int width) {
    Matrix& resultRef = *CreateMatrixPointer(hieght, width);
    return resultRef;
}
Matrix* MatrixMemoryManager::CreateMatrixPointer(Matrix& source, bool isDeepCopy) {
    // Tạo ra một ma trận rỗng bằng Default Constructor để tránh cấp phát thừa vùng nhớ
    Matrix* createResult = new Matrix();
    createResult->cublasHandle = handle; // Gắn handle của Manager vào

    // Thực hiện Copy dữ liệu
    if (isDeepCopy) {
        DeepCopy(*createResult, source);
    } else {
        ShallowCopy(*createResult, source);
    }

    // Đưa ma trận mới (đã copy xong giá trị) vào danh sách quản lý
    matrixList.push_back(createResult);

    return createResult;
}

Matrix* MatrixMemoryManager::CreateMatrixPointer(const int h,const int w) {
    Matrix* createResult = new Matrix(h, w, handle);
    matrixList.push_back(createResult);
    return createResult;
}

void MatrixMemoryManager::removeAllMatrix() {
    matrixList.clear();
}

MatrixMemoryManager::~MatrixMemoryManager() = default;
