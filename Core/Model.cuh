//
// Created by tommydatlc on 3/25/26.
//

#ifndef OPTIMIZESDF_MODEL_H
#define OPTIMIZESDF_MODEL_H
#include <string>
#include <vector>

// Include các module nội bộ
#include "ClipSpaceConversion.hpp"
#include "Matrix.cuh"


class Model {
private:
    std::vector<std::array<double, 3> > vertex;
    std::vector<std::array<size_t, 3> > faces;
    Matrix *cacheVertexMatrix = nullptr;
    Matrix *cacheIndicesMatrix = nullptr;
    Matrix *VertexNormalMatrix = nullptr;
    Matrix *FaceNormalMatrix = nullptr;
    std::vector<double> vertexAttributes;
public:
    Model(std::string filename); //Read model using ReadFromObjFile
    void ReadFromObjFile(std::string filename); // Read file obj from file name
    // Create an new matrix by using Matrix &MatrixMemoryManager::CreateMatrix(width,height)
    Matrix &GetVertexMatrix(); // loading the matrix vertex (one vertex per column)
    Matrix &GetVertexIndicesMatrix();

    // loading the the index as adjection matrix, save into cache, if cache not null, return the cache instead
    void SetVertexMatrix(Matrix &newVertex);

    // replace the cache matrix, and replace the Vertices vector using the newMatrix
    void SaveObjFile(std::string filename); // save object as an obj file
    void AddToScene(std::string name = "model", bool displayNormal = true); // Preview by using Polyscope
    void UpdateNormal();
    void AddHeatMapVertexForPreviewEngine(int vertexIndex, double attribute);
    void ToClipSpace(ViewData &V, PerspectiveCameraData &P);
};


#endif //OPTIMIZESDF_MODEL_H