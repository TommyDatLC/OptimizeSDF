#ifndef OPTIMIZESDF_MODEL_H
#define OPTIMIZESDF_MODEL_H
#include <string>
#include <vector>
#include <array>
#include "Matrix.cuh"

class Model {
private:
    std::vector<std::array<double, 3>> vertex;
    std::vector<std::array<size_t, 3>> faces;

    // Đã chuyển Matrix thành Template chuẩn, phân chia rõ ràng float (tọa độ) và uint (index)
    Matrix<float> *cacheVertexMatrix = nullptr;
    Matrix<unsigned int> *cacheIndicesMatrix = nullptr;
    Matrix<float> *vertexNormalMatrix = nullptr;
    Matrix<float> *faceNormalMatrix = nullptr;

    std::vector<double> vertexAttributes;
    bool showHeatMap = false;

public:
    Model(std::string filename);
    ~Model();
    void ReadFromObjFile(std::string filename);

    Matrix<float> &GetVertexMatrix();
    Matrix<unsigned int> &GetVertexIndicesMatrix();
    Matrix<float> &GetVertexNormalMatrix();

    void SetVertexMatrix(Matrix<float> &newVertex);
    void SaveObjFile(std::string filename);
    void AddToScene(std::string name = "model", bool displayNormal = true);
    void UpdateNormal(cudaStream_t stream = 0);
    void AddHeatMapVertexForPreviewEngine(int index, double value);
    const std::vector<double>& GetVertexAttributes() const;
    int GetVertexCount() const { return vertex.size(); }
    int GetFaceCount() const { return faces.size(); }
    void SetShowHeatMap(bool show);
};

#endif //OPTIMIZESDF_MODEL_H