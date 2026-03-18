#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>

#include "MatrixMemoryManager.cpp"
#include "Matrix.cpp"
int main() {
    auto matrixMemMang = MatrixMemoryManager();
    Matrix& m1 = *matrixMemMang.CreateMatrix(4,3);
    m1.Set(0,0,-1);
    m1.Set(0,1,1);
    m1.Set(0,2,-1);

    m1.Set(1,0,5);
    m1.Set(1,1,2);
    m1.Set(1,2,-5);

    m1.Set(2,0,6);
    m1.Set(2,1,-5);
    m1.Set(2,2,1);

    m1.Set(3,0,-5);
    m1.Set(3,1,-6);
    m1.Set(3,2,0);

    Matrix& m2 = *matrixMemMang.CreateMatrix(3,2);
    m2.Set(0,0,6);
    m2.Set(0,1,5);

    m2.Set(1,0,5);
    m2.Set(1,1,-6);

    m2.Set(2,0,6);
    m2.Set(2,1,0);
    m1.CopyToDevice();
    m2.CopyToDevice();
    Matrix m3 = m2 * m1;
    m3.PrintOnGPU();

}
