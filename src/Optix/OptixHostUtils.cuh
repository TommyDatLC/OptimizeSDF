#ifndef OPTIX_HOST_UTILS_CUH
#define OPTIX_HOST_UTILS_CUH

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// -----------------------------------------------------------------------------
// STRUCT ĐỒNG BỘ VỚI GPU
// Căn chỉnh 8 bytes chuẩn xác. Không xài x, y, z, w
// -----------------------------------------------------------------------------
struct alignas(8) Params {
    float3* vertices;
    float3* normals;
    float* outputDistances;
    float* outputWeights;
    int* validCounts;
    OptixTraversableHandle bvhHandle;
    int raysPerPoint;
    float coneAngleRad;
    float2 hammersleyUVs[128];
};

template <typename T>
struct SbtRecord {
    alignas( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<int> RaygenRecord;
typedef SbtRecord<int> MissRecord;
typedef SbtRecord<int> HitgroupRecord;

#define OPTIX_CHECK( call ) { OptixResult res = call; if( res != OPTIX_SUCCESS ) { fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); exit( 2 ); } }
#define CUDA_CHECK( call ) { cudaError_t error = call; if( error != cudaSuccess ) { fprintf( stderr, "CUDA call (%s) failed with code %d (line %d)\n", #call, error, __LINE__ ); exit( 2 ); } }

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */) { 
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n"; 
}

#endif // OPTIX_HOST_UTILS_CUH
