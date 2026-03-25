//
// Created by tommydatlc on 3/17/26.
//

#include "MatrixMemoryManager.cuh"
#include <cublas_v2.h>
#include <memory>

#include "Matrix.cuh"
MatrixMemoryManager::MatrixMemoryManager()  {
    cublasCreate(&handle);
}

Matrix &MatrixMemoryManager::CreateMatrix(const int hieght, const int width) {
    Matrix& resultRef = *CreateMatrixPointer(hieght, width);
    return resultRef;
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
