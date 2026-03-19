#ifndef CLIPPING_SPACE
#define CLIPPING_SPACE
#include "MatrixMemoryManager.h"
#include "mathHelper.cpp"
struct TransformData {
    float3 Translate;
    float3 Rotation; // Giả định là Euler angles (Pitch, Yaw, Roll) theo Radian
    float3 Size;     // Scale
    Matrix *cache = 0;

    Matrix& GetMatrix() {
        if (cache)
            return *cache;
        Matrix& mat = matrixMemMang.CreateMatrix(4, 4);

        // Tính toán các giá trị lượng giác cho Rotation
        float cx = cosf(Rotation.x); float sx = sinf(Rotation.x);
        float cy = cosf(Rotation.y); float sy = sinf(Rotation.y);
        float cz = cosf(Rotation.z); float sz = sinf(Rotation.z);

        // Khởi tạo ma trận T * R * S (Dữ liệu lưu theo Cột - Column Major)
        // Cột 0
        mat.Set(0, 0, (cy * cz) * Size.x);
        mat.Set(1, 0, (cy * sz) * Size.x);
        mat.Set(2, 0, (-sy) * Size.x);
        mat.Set(3, 0, 0.0f);

        // Cột 1
        mat.Set(0, 1, (sx * sy * cz - cx * sz) * Size.y);
        mat.Set(1, 1, (sx * sy * sz + cx * cz) * Size.y);
        mat.Set(2, 1, (sx * cy) * Size.y);
        mat.Set(3, 1, 0.0f);

        // Cột 2
        mat.Set(0, 2, (cx * sy * cz + sx * sz) * Size.z);
        mat.Set(1, 2, (cx * sy * sz - sx * cz) * Size.z);
        mat.Set(2, 2, (cx * cy) * Size.z);
        mat.Set(3, 2, 0.0f);

        // Cột 3 (Translation)
        mat.Set(0, 3, Translate.x);
        mat.Set(1, 3, Translate.y);
        mat.Set(2, 3, Translate.z);
        mat.Set(3, 3, 1.0f);


        mat.CopyToDevice();
        cache = &mat;
        return mat;
    }
};

struct PerspectiveCameraData {
    float FOV;  // Dạng Radian
    float Near;
    float Far;
    Matrix *cache = nullptr;
    // Bổ sung AspectRatio (chiều rộng / chiều cao của viewport). Mặc định là 16:9
    float AspectRatio = 16.0f / 9.0f;

    Matrix& GetMatrix() {
        if (cache != nullptr)
            return *cache;
        Matrix& mat = matrixMemMang.CreateMatrix(4, 4);

        // Công thức chuẩn Perspective Projection
        float f = 1.0f / tanf(FOV * 0.5f);

        mat.Set(0, 0, f / AspectRatio);
        mat.Set(1, 1, f);
        mat.Set(2, 2, (Far + Near) / (Near - Far));
        mat.Set(3, 2, -1.0f);
        mat.Set(2, 3, (2.0f * Far * Near) / (Near - Far));

        mat.CopyToDevice();
        cache = &mat;
        return mat;
    }
};

struct ViewData {
    float3 Position; // ĐÃ BỔ SUNG: Bắt buộc phải có vị trí camera
    float3 LookAtX;  // Hiểu là Target (Điểm camera nhìn vào)
    float3 WorldUp;
    float3 CameraUp; // Trục Y hướng lên của Camera
    Matrix *cache = 0;
    Matrix& GetMatrix() {
        if (cache)
            return *cache;
        Matrix& mat = matrixMemMang.CreateMatrix(4, 4);

        // Helper functions nội bộ xử lý vector (do CUDA float3 không có sẵn toán tử)
        // Tính toán các trục của Camera (Right-Handed Coordinate System)
        float3 zaxis = normalize(sub(Position, LookAtX)); // Forward (âm)
        float3 xaxis = normalize(cross(CameraUp, zaxis)); // Right
        float3 yaxis = cross(zaxis, xaxis);               // Up thực tế

        // Xây dựng View Matrix (LookAt Matrix)
        mat.Set(0, 0, xaxis.x);
        mat.Set(0, 1, xaxis.y);
        mat.Set(0, 2, xaxis.z);
        mat.Set(0, 3, -dot(xaxis, Position));

        mat.Set(1, 0, yaxis.x);
        mat.Set(1, 1, yaxis.y);
        mat.Set(1, 2, yaxis.z);
        mat.Set(1, 3, -dot(yaxis, Position));

        mat.Set(2, 0, zaxis.x);
        mat.Set(2, 1, zaxis.y);
        mat.Set(2, 2, zaxis.z);
        mat.Set(2, 3, -dot(zaxis, Position));

        mat.Set(3, 3, 1.0f);

        mat.CopyToDevice();
        cache = &mat;
        return mat;
    }
};
Matrix& ClippingSpaceConversion(TransformData M, ViewData V, PerspectiveCameraData P,const Matrix& point_list) {
    Matrix& MVP = M.GetMatrix() * V.GetMatrix() * P.GetMatrix() * point_list;
    //std::cout << MVP.Height << "x" << MVP.Width << std::endl;
   // std::cout << "MVP matrix: ";
   //  MVP.PrintOnGPU();
    return MVP;
}
#endif