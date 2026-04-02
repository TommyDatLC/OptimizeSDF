#include "MatrixMemoryManager.cuh"

#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include "ClipSpaceConversion.hpp"
#include "Matrix.cuh"
#include "Model.h"
#include "ModelHelper.cu"
#include "cmake-build-debug/_deps/polyscope-src/include/polyscope/polyscope.h"

//
// void TestMatrix() {
//     // --- TEST CASE 1 ---
//     printf("--- Test Case 1 ---\n");
//     Matrix& A1 = matrixMemMang.CreateMatrix(2, 2);
//     A1.Set(0, 0, 0);  A1.Set(0, 1, 2);
//     A1.Set(1, 0, -2); A1.Set(1, 1, -5);
//     A1.CopyToDevice();
//
//     Matrix& B1 = matrixMemMang.CreateMatrix(2, 2);
//     B1.Set(0, 0, 6); B1.Set(0, 1, -6);
//     B1.Set(1, 0, 3); B1.Set(1, 1, 0);
//     B1.CopyToDevice();
//
//     Matrix& Ans1 = A1 * B1;
//     Ans1.PrintOnGPU(); // Kỳ vọng: [6, 0], [-27, 12]
//
//     // --- TEST CASE 2 ---
//     printf("\n--- Test Case 2 ---\n");
//     Matrix& A2 = matrixMemMang.CreateMatrix(2, 1);
//     A2.Set(0, 0, 6);
//     A2.Set(1, 0, -3);
//     A2.CopyToDevice();
//
//     Matrix& B2 = matrixMemMang.CreateMatrix(1, 2);
//     B2.Set(0, 0, -5); B2.Set(0, 1, 4);
//     B2.CopyToDevice();
//
//     Matrix& Ans2 = A2 * B2;
//     Ans2.PrintOnGPU(); // Kỳ vọng: [-30, 24], [15, -12]
//
//     // --- TEST CASE 3 ---
//     printf("\n--- Test Case 3 ---\n");
//     Matrix& A3 = matrixMemMang.CreateMatrix(2, 2);
//     A3.Set(0, 0, -5); A3.Set(0, 1, -5);
//     A3.Set(1, 0, -1); A3.Set(1, 1, 2);
//     A3.CopyToDevice();
//
//     Matrix& B3 = matrixMemMang.CreateMatrix(2, 2);
//     B3.Set(0, 0, -2); B3.Set(0, 1, -3);
//     B3.Set(1, 0, 3);  B3.Set(1, 1, 5);
//     B3.CopyToDevice();
//
//     Matrix& Ans3 = A3 * B3;
//     Ans3.PrintOnGPU(); // Kỳ vọng: [-5, -10], [8, 13]
//
//     // --- TEST CASE 4 ---
//     printf("\n--- Test Case 4 ---\n");
//     Matrix& A4 = matrixMemMang.CreateMatrix(2, 2);
//     A4.Set(0, 0, -3); A4.Set(0, 1, 5);
//     A4.Set(1, 0, -2); A4.Set(1, 1, 1);
//     A4.CopyToDevice();
//
//     Matrix& B4 = matrixMemMang.CreateMatrix(2, 2);
//     B4.Set(0, 0, 6); B4.Set(0, 1, -2);
//     B4.Set(1, 0, 1); B4.Set(1, 1, -5);
//     B4.CopyToDevice();
//
//     Matrix& Ans4 = A4 * B4;
//     Ans4.PrintOnGPU(); // Kỳ vọng: [-13, -19], [-11, -1]
// }
int main() {

    polyscope::init();
    // TestMatrix();
   //  Matrix& m1 = matrixMemMang.CreateMatrix(4,2);
   //  m1.Set(0,0,1);
   //  m1.Set(1,0,2);
   //  m1.Set(3,0,3);
   //  m1.Set(2,0,4);
   //  m1.CopyToDevice();
   //  // TransformData t_data;
   //  // t_data.Size = float3(1,1,1);
   //  // t_data.Translate = float3(3,4,5);
   //  // t_data.Rotation = float3(1,2,3);
   //  ViewData v_data;
   //  v_data.CameraUp = float3(3,1,0);
   //  v_data.Position = float3(2,1,1);
   //  v_data.LookAtX = float3(1,10,3);
   //  PerspectiveCameraData p_data;
   //  p_data.Far = 30;
   //  p_data.Near = 10;
   //  p_data.FOV = 40;
   //  Model model = Model("360.obj");// nho ghi absolute path vo
   //  model.AddToScene("model2");
   // // model.CopyToDevice();
   //  //model.Print();
   //  std::cout << '\n';
   //  //m1.PrintOnGPU();
   //  Matrix& modelVertMatrix = model.GetVertexMatrix();
   //  modelVertMatrix.CopyToDevice();
   //  Matrix& FinalResult = ClippingSpaceConversion(v_data,p_data,modelVertMatrix);
   //  FinalResult.CopyToHost();
   //  model.SetVertexMatrix(FinalResult);
   //  model.AddToScene("model1");
   //  polyscope::show();
   //  FinalResult.PrintOnGPU();
}
