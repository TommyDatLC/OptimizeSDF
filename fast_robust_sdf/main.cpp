// ==================================================================
// Fast and Robust Shape Diameter Function  (Chen et al. 2018)
//   C++ reimplementation of the offset-surface SDF algorithm
//   with detailed per-step timing and logging
//
// Reuses OBJ loading pattern from the OptiX SDF project's
//   Core/Model.cu  and math from Core/MathHelper.cu
// ==================================================================

#include "sdf_offset.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <numeric>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

using Clock = std::chrono::high_resolution_clock;

// ============================================================
// File output -- save SDF per vertex  (same CSV format as PyMeshLab output)
//   vertexID, sdfValue
// ============================================================
static bool saveSdfToFile(const std::string& filename, const SdfOffset& sdf)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return false;

    ofs << "# Fast Robust SDF  (offset-surface method)\n";
    ofs << "# vertexID, sdfValue\n";
    for (size_t i = 0; i < sdf.sdfValues.size(); ++i) {
        ofs << i << "," << std::setprecision(10) << sdf.sdfValues[i] << "\n";
    }
    return true;
}

// ============================================================
// Log summary table
// ============================================================
static void printSummary(const SdfOffset& sdf, double totalSeconds)
{
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  PERFORMANCE SUMMARY\n";
    std::cout << "============================================================\n";
    std::cout << "  Vertices:          " << sdf.vertices.size()   << "\n";
    std::cout << "  Triangles:         " << sdf.triangles.size()  << "\n";
    std::cout << "  Offset components: " << 7 * sdf.triangles.size() << "\n";
    std::cout << "  Offset radius:     " << std::setprecision(6) << sdf.offsetRadius << "\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  Step 1 - Load OBJ:           " << std::setw(10) << std::setprecision(4) << sdf.timeLoad    << " s\n";
    std::cout << "  Step 2 - Vertex normals:      " << std::setw(10) << std::setprecision(4) << sdf.timeNormals << " s\n";
    std::cout << "  Step 3 - Offset components:   " << std::setw(10) << std::setprecision(4) << sdf.timeOffset  << " s\n";
    std::cout << "  Step 4 - OBB tree build:      " << std::setw(10) << std::setprecision(4) << sdf.timeTree    << " s\n";
    std::cout << "  Step 5 - SDF computation:     " << std::setw(10) << std::setprecision(4) << sdf.timeSdf     << " s\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  Total wall-clock:             " << std::setw(10) << std::setprecision(4) << totalSeconds   << " s\n";
    std::cout << "============================================================\n";
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[])
{
    // ---- configuration (reuses pattern from the OptiX project's main.cu) ----
    std::string modelDir = "Model";
    bool showVisualization = true;

    // Parse command line
    if (argc >= 2) modelDir = argv[1];
    if (argc >= 3) {
        std::string arg2 = argv[2];
        if (arg2 == "0" || arg2 == "false" || arg2 == "no")
            showVisualization = false;
    }

    // Discover OBJ files (same pattern as original main.cu)
    std::vector<std::string> objFiles;
    for (auto& entry : std::filesystem::directory_iterator(modelDir)) {
        if (entry.path().extension() == ".obj")
            objFiles.push_back(entry.path().string());
    }
    std::sort(objFiles.begin(), objFiles.end());

    if (objFiles.empty()) {
        std::cerr << "No .obj files found in: " << modelDir << "\n";
        return 1;
    }

    std::cout << "================================================================\n";
    std::cout << "  Fast & Robust Shape Diameter Function  (Chen et al. 2018)\n";
    std::cout << "  Offset-surface based SDF  -- CPU reimplementation\n";
    std::cout << "================================================================\n";
    std::cout << "Found " << objFiles.size() << " OBJ files in '" << modelDir << "'\n\n";

    // ---- polyscope init (same as original project) ----
    polyscope::options::automaticallyComputeSceneExtents = false;
    polyscope::init();

    int fileIdx = 0;
    for (const auto& objPath : objFiles) {
        fileIdx++;
        std::cout << "================================================================\n";
        std::cout << "  [" << fileIdx << "/" << objFiles.size() << "] " << objPath << "\n";
        std::cout << "================================================================\n";

        auto wallStart = Clock::now();

        SdfOffset sdf;

        // Step 1: Load mesh  (reuses pattern from Model::ReadFromObjFile)
        sdf.loadObj(objPath);

        // Step 2: Compute vertex normals  (reuses pattern from ModelHelper.cu)
        sdf.computeVertexNormals();

        // Step 3: Build offset surface components  (paper Section 4)
        sdf.buildOffsetComponents();

        // Step 4: Build OBB tree  (paper Section 4.2)
        sdf.buildObbTree();

        // Step 5: Compute SDF  (paper Algorithm 1)
        sdf.computeSdf();

        double totalSeconds = std::chrono::duration<double>(Clock::now() - wallStart).count();

        // Print performance summary
        printSummary(sdf, totalSeconds);

        // Save results to CSV file  (same format as PyMeshLab output)
        std::string baseName = std::filesystem::path(objPath).stem().string();
        std::string sdfFile  = baseName + "_offset_sdf.csv";
        if (saveSdfToFile(sdfFile, sdf)) {
            std::cout << "Saved SDF to: " << sdfFile << "\n";
        } else {
            std::cerr << "Failed to save: " << sdfFile << "\n";
        }

        // ---- Visualization with polyscope (same as original AddToScene) ----
        if (showVisualization && !sdf.vertices.empty() && !sdf.triangles.empty()) {
            // Convert to polyscope-compatible format
            std::vector<std::array<double, 3>> psVerts(sdf.vertices.size());
            for (size_t i = 0; i < sdf.vertices.size(); ++i) {
                psVerts[i] = {sdf.vertices[i].x, sdf.vertices[i].y, sdf.vertices[i].z};
            }
            std::vector<std::array<size_t, 3>> psFaces(sdf.triangles.size());
            for (size_t i = 0; i < sdf.triangles.size(); ++i) {
                psFaces[i] = {sdf.triangles[i].i0, sdf.triangles[i].i1, sdf.triangles[i].i2};
            }

            auto* psMesh = polyscope::registerSurfaceMesh(baseName, psVerts, psFaces);

            // Add SDF heat map  (same as Model::AddToScene)
            auto* sdfQ = psMesh->addVertexScalarQuantity("SDF (offset)", sdf.sdfValues);
            sdfQ->setEnabled(true);
            sdfQ->setColorMap("turbo");
            sdfQ->resetMapRange();
        }

        std::cout << "\n";
    }

    // Launch polyscope viewer
    if (showVisualization) {
        std::cout << "Opening Polyscope viewer... Close the window to exit.\n";
        polyscope::show();
    }

    return 0;
}
