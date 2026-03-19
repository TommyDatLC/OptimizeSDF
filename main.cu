#include "MatrixMemoryManager.cpp"

#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include "ClipSpaceConversion.cpp"
#include "Matrix.cpp"
int main() {
    Matrix& m1 = matrixMemMang.CreateMatrix(4,2);
    m1.Set(1,0,1);
    m1.Set(1,0,2);
    m1.Set(1,0,3);
    m1.Set(1,0,4);
    m1.CopyToDevice();
    TransformData t_data;
    t_data.Size = float3(1,1,1);
    t_data.Translate = float3(3,4,5);
    t_data.Rotation = float3(1,2,3);
    ViewData v_data;
    v_data.CameraUp = float3(0,1,0);
    v_data.Position = float3(2,1,1);
    v_data.LookAtX = float3(0,10,3);
    PerspectiveCameraData p_data;
    p_data.Far = 30;
    p_data.Near = 10;
    p_data.FOV = 50;

    Matrix& FinalResult = ClippingSpaceConversion(t_data,v_data,p_data,m1);
    FinalResult.PrintOnGPU();
}
