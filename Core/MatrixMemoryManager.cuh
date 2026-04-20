//
// Created by tommydatlc on 3/17/26.
//
#ifndef CUBLASAPI
#define CUBLASAPI
#endif
#ifndef OPTIMIZESDF_MATRIXMEMORYMANAGER_H
#define OPTIMIZESDF_MATRIXMEMORYMANAGER_H
#include <cublas_api.h>
#include <memory>
#include <vector>
#include "Matrix.cuh"
class MatrixMemoryManager {

        public:
        MatrixMemoryManager();

        Matrix &CreateMatrix(int h, int w);

        Matrix *CreateMatrixPointer(Matrix &source, bool isDeepCopy);

        Matrix* CreateMatrixPointer(int h, int w);
        void removeAllMatrix();
        cublasHandle_t handle = nullptr;
        ~MatrixMemoryManager();
    // ----- REMEMBER TO PUT ANOTHER POINTER HERE REFERENCE COPY
        void ShallowCopy(Matrix &dest, Matrix &src);

        void DeepCopy(Matrix& source,const Matrix& destination);// FULL COPY;
    };
inline auto matrixMemMang = MatrixMemoryManager();

#endif //OPTIMIZESDF_MATRIXMEMORYMANAGER_H