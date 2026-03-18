//
// Created by tommydatlc on 3/17/26.
//

#ifndef OPTIMIZESDF_MATRIXMEMORYMANAGER_H
#define OPTIMIZESDF_MATRIXMEMORYMANAGER_H
#include <cublas_api.h>
#include <memory>
#include <vector>
#include "Matrix.h"
class MatrixMemoryManager {

    std::vector<Matrix*> matrixList;
    public:
    MatrixMemoryManager();
    Matrix* CreateMatrix(int h,int w);
    void removeAllMatrix();
    cublasHandle_t handle = nullptr;
    ~MatrixMemoryManager();
};


#endif //OPTIMIZESDF_MATRIXMEMORYMANAGER_H