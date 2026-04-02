//
// Created by tommydatlc on 4/1/26.
//
#ifndef ModelHelper
#define ModelHelper
#include <device_launch_parameters.h>
#include "Matrix.cuh"
#include "mathHelper.cu"

inline __global__ void GPUNormalCaculation(Matrix& vertexList // one vertex is one column height 4
    ,Matrix& indicesList // one indces is one column height 3
    ,Matrix& facesNormalList, // output for face normal
    Matrix& pointsNormalList // output for vertex normal
    )
{
    // looping by index ID;
    // get the vertex 1 from indicesList
    // get vertex 2 from indices List
    // get vertex 3 from indeces List
    // do cross product betwwen vector vertex 2 - vertex 1 and vertex 3 - vertex 1 for caculating the normal (cross product is in mathHelper.cu)
    // put that into normal point list vectorlist
    // update the normal vector of vertex by doing some normal
}
#endif
