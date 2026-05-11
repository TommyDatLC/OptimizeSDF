# **Optimizing Shape Diameter Function using HPC**
The Shape Diameter Function (SDF) is a scalar function defined on a 3D mesh surface that measures the local volume thickness (diameter) around each point, providing a pose-oblivious, informative signature for shape analysis. It calculates an average ray intersection distance, heavily used in mesh segmentation and skeletonization to identify distinct, similar components.


In this project, we try to optimize SDF function by using CUDA C++, and compare it with the previous work: Parallelization of Shape Diameter Function Computation using
OpenCL
The link of this work: https://old.cescg.org/CESCG-2014/papers/Kamenicky-Parallelization_of_Shape_Diameter_Function_Computation_using_OpenCL.pdf


The Speed improvement: 

## Performance Benchmark: OpenCL vs. NVIDIA OptiX

| Model Name | Vertices | OpenCL (s) | OptiX (s) | Faster API 🏆 |
| :--- | :---: | :---: | :---: | :--- |
| Crate.obj | 386 | 0.00258 | 0.06316 | ⚡ OpenCL (2448%) |
| Earth.obj | 952 | 0.00781 | 0.06176 | ⚡ OpenCL (791%) |
| 360.obj | 2,200 | 0.02407 | 0.06168 | ⚡ OpenCL (256%) |
| 9.obj | 2,639 | 0.02937 | 0.06154 | ⚡ OpenCL (210%) |
| 400.obj | 3,703 | 0.04541 | 0.09141 | ⚡ OpenCL (201%) |
| 76.obj | 5,923 | 0.07216 | 0.07522 | ⚡ OpenCL (104%) |
| 181.obj | 7,242 | 0.08811 | 0.06035 | 🟢 OptiX (146%) |
| 118.obj | 9,153 | 0.10206 | 0.06512 | 🟢 OptiX (157%) |
| 368.obj | 11,202 | 0.14285 | 0.06672 | 🟢 OptiX (214%) |
| 369.obj | 13,606 | 0.17463 | 0.06319 | 🟢 OptiX (276%) |
| 112.obj | 13,628 | 0.15290 | 0.06154 | 🟢 OptiX (248%) |
| 158.obj | 14,587 | 0.15723 | 0.06227 | 🟢 OptiX (252%) |
| 371.obj | 14,599 | 0.19217 | 0.06237 | 🟢 OptiX (308%) |
| Leaf.obj | 24,866 | 0.37753 | 0.06418 | 🟢 OptiX (588%) |
| xyzrgb_dragon | 500,079 | 1.11001 | 0.24432 | 🟢 OptiX (4500%) |

## Technical Summary
- **OpenCL** excels with low-poly models due to minimal overhead.
- **OptiX** dominates high-poly scenes by leveraging hardware acceleration.


## 2. Visual Quality Comparison

The following table demonstrates that despite the performance differences between the APIs, the visual output remains consistent across both rendering backends.

| Model | NVIDIA OptiX | OpenCL |
| :--- | :---: | :---: |
| **Leaft**<br>* | <img alt="Screenshot From 2026-05-11 08-47-21" src="https://github.com/user-attachments/assets/d6f3c9bc-2b6d-47df-9361-78406d5db55b" />| <img alt="image" src="https://github.com/user-attachments/assets/e1fe8ca9-95c1-4d26-ab2b-53b7e776e75d" />  |
| **Dragon**<br>* | <img src="https://github.com/user-attachments/assets/a78a2a2f-85a8-438c-8ee6-1d61890c58ac" width="100%" alt="Optix Dragon"> | <img src="https://github.com/user-attachments/assets/5ff27ab3-062c-4525-bf3b-9c73e8c4f48c" width="100%" alt="OpenCL Dragon"> |
| **Hand**<br>* | <img src="https://github.com/user-attachments/assets/5c8f0d48-7f77-42dc-8967-85adb96dffba" width="100%" alt="Optix Crate"> | <img src="https://github.com/user-attachments/assets/af490257-b130-43ff-bb35-60d8eff5c736" width="100%" alt="OpenCL Crate"> |

## Setup and Run

> ⚠️ **CAUTION: Linux Only**
> This project is strictly designed for **Linux environments** and will not compile or run correctly on Windows or macOS.

### Prerequisites
To successfully build and execute this project, ensure the following tools are installed on your system:
* **`g++`**: The GNU C++ compiler for building standard C++ source files.
* **`nvcc`**: The NVIDIA CUDA Compiler toolkit, required for compiling the OptiX/CUDA components.
* **`CLion`**: The recommended IDE to load the CMake project, resolve dependencies, and execute the builds properly.

