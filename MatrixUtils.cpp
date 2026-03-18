//
// Created by tommydatlc on 3/18/26.
//
#include <stdio.h>
#ifndef OPTIMIZESDF_MATRIXUTILS_H
#define OPTIMIZESDF_MATRIXUTILS_H
// ----------------------- UTILITIES FUNCTION------------------------
__host__ __device__ void printMemory(float* mem_region,int height,int width) {
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
#endif //OPTIMIZESDF_MATRIXUTILS_H