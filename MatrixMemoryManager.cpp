//
// Created by tommydatlc on 3/17/26.
//

#include "MatrixMemoryManager.h"
#include <cublas_v2.h>
#include <memory>

#include "Matrix.h"
MatrixMemoryManager::MatrixMemoryManager()  {
    cublasCreate(&handle);
}

Matrix* MatrixMemoryManager::CreateMatrix(const int hieght,const int width) {
    Matrix* createResult = new Matrix(hieght, width, handle);
    matrixList.push_back(createResult);
    return createResult;
}

void MatrixMemoryManager::removeAllMatrix() {
    matrixList.clear();
}

MatrixMemoryManager::~MatrixMemoryManager() = default;
