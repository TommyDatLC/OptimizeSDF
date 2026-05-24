//
// Created by tommydatlc on 3/18/26.
//
#ifndef OPTIMIZESDF_MATRIXUTILS_H
#define OPTIMIZESDF_MATRIXUTILS_H
#include <stdio.h>
#include <iostream>
#include "MatrixMemoryManager.cu"


// ----------------------- UTILITIES FUNCTION------------------------
__host__ __device__ void printMemory(float* mem_region,int height,int width) {
   // printf("matrix w h: %dx%d\n",height,width);
    for (int h = 0; h < height; h++) {         // Lặp qua từng hàng
        for (int w = 0; w < width; w++) {     // Lặp qua từng cột trong hàng đó
            // Sử dụng hàm Get bạn đã viết hoặc truy cập trực tiếp
            printf("%.2f\t", mem_region[w * height + h]);
        }
        printf("\n"); // Xuống dòng sau khi in hết một hàng
    }
}
__global__ void printOnGPUKernel(float* mem_region,int height,int width) {
    printMemory(mem_region,height,width);
}



//
// Matrix& LoadVertexFromObj(std::string objFile) {
//     std::vector<float3> vertices;
//     FILE *res = freopen(objFile.c_str(), "r",stdin);
//     if (res == nullptr)
//         std::__throw_runtime_error("File cannot be opened");
//     char first_char = -1;
//     int i = 0;
//     while (std::cin >> first_char)
//      {
//         if (first_char != 'v') {
//             std::string dummy;
//             std::getline(std::cin, dummy);
//             continue;
//         }
//         float3 new_vertex;
//         std::cin >> new_vertex.x >> new_vertex.y >> new_vertex.z;
//         // std::cout << i << " : " << new_vertex.x << " " << new_vertex.y << " " << new_vertex.z << std::endl;
//         vertices.push_back(new_vertex);
//     }
//     //std::cout << "Number of vertices: " << vertices.size() << std::endl;
//     Matrix& model = matrixMemMang.CreateMatrix(4,vertices.size());
//     for (int i = 0;i < vertices.size();i++) {
//         model.Set(0,i,vertices[i].x);
//         model.Set(1,i,vertices[i].y);
//         model.Set(2,i,vertices[i].z);
//         model.Set(3,i,1);
//     }
//     return model;
// }
#endif //OPTIMIZESDF_MATRIXUTILS_H