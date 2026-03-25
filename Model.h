//
// Created by tommydatlc on 3/25/26.
//

#ifndef OPTIMIZESDF_MODEL_H
#define OPTIMIZESDF_MODEL_H
#include <string>
#include <vector>

// Include các module nội bộ
#include "Matrix.cuh"
#include "MatrixMemoryManager.cuh"

class Model {
private:
    Matrix* cacheVertexMatrix;
    Matrix* cacheIndicesMatrix;

    public:
    Model(std::string filename); //Read model using ReadFromObjFile
    std::vector<float3> Vertices;
    std::vector<float3> VertexIndices;
    void ReadFromObjFile(std::string filename);// Read file obj from file name
    // Create an new matrix by using Matrix &MatrixMemoryManager::CreateMatrix(width,height)
    Matrix& GetVertexMatrix();// loading the matrix vertex (one vertex per column)
    Matrix& GetVertexIndicesMatrix();// loading the the index as adjection matrix, save into cache, if cache not null, return the cache instead
    void SetVertexMatrix(Matrix &newVertex);// replace the cache matrix, and replace the Vertices vector using the newMatrix
    void SaveObjFile(std::string filename); // save object as an obj file
    void Preview(); // Preview by using Polyscope
};


#endif //OPTIMIZESDF_MODEL_H