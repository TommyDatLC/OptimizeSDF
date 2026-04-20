#include "Core/MatrixMemoryManager.cuh"
#include "NormalSDFOptix/SDFMain.cpp"
#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include "Core/ClipSpaceConversion.hpp"
#include "Core/Matrix.cuh"
#include "Core/Model.cuh"
#include "Core/ModelHelper.cu"
#include "cmake-build-debug/_deps/polyscope-src/include/polyscope/polyscope.h"
#include "OptiX/optix.h"
#include "OptiX/optix_stubs.h"
#include <optix_function_table_definition.h>
int main() {
    optixInit();
    polyscope::init();
     // TestMatrix();
     // Matrix& m1 = matrixMemMang.CreateMatrix(4,2);
     // m1.Set(0,0,1);
     // m1.Set(1,0,2);
     // m1.Set(3,0,3);
     // m1.Set(2,0,4);
     // m1.CopyToDevice();
     // TransformData t_data;
     // t_data.Size = float3(1,1,1);
     // t_data.Translate = float3(3,4,5);
     // t_data.Rotation = float3(1,2,3);
        ViewData v_data;
        v_data.CameraUp = float3(3,1,0);
        v_data.Position = float3(2,1,1);
        v_data.LookAtX = float3(1,10,3);
        PerspectiveCameraData p_data;
        p_data.Far = 30;
        p_data.Near = 10;
        p_data.FOV = 40;
        Model model = Model("Model/360.obj");// nho ghi absolute path vo
        model.UpdateNormal();
        model.ToClipSpace(v_data,p_data);
        model.AddToScene();
        polyscope::show();
        //CaculatingSDFUsingOptix(model);
}
