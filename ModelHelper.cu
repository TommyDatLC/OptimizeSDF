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
    int id_indicesList = blockDim.x * blockIdx.x + threadIdx.x;
    // get the vertex 1 from indicesList
    int vertex1ID = indicesList.GetDevice(id_indicesList,0 );
    // get vertex 2 from indices List
    int vertex2ID = indicesList.GetDevice(id_indicesList,1 );
    // get vertex 3 from indeces List
    int vertex3ID = indicesList.GetDevice(id_indicesList,2 );
    float3 vert1 = float3(vertexList.GetDevice(vertex1ID,0),vertexList.GetDevice(vertex1ID,1),vertexList.GetDevice(vertex1ID,2));
    float3 vert2 = float3(vertexList.GetDevice(vertex2ID,0),vertexList.GetDevice(vertex2ID,1),vertexList.GetDevice(vertex2ID,2));
    float3 vert3 = float3(vertexList.GetDevice(vertex3ID,0),vertexList.GetDevice(vertex3ID,1),vertexList.GetDevice(vertex3ID,2));
    // do cross product betwwen vector vertex 2 - vertex 1 and vertex 3 - vertex 1 for caculating the normal (cross product is in mathHelper.cu)
    float3 face_normal = cross(vert2 - vert1,vert3 - vert1);

    facesNormalList.SetDevice(id_indicesList,0,face_normal.x);
    facesNormalList.SetDevice(id_indicesList,1,face_normal.y);
    facesNormalList.SetDevice(id_indicesList,2,face_normal.z);
    // put that into normal point list vectorlist


    // update the normal vector of vertex by doing some normal
}
#endif
