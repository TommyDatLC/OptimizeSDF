**Optimizing Shape Diameter FUnction using HPC**
The Shape Diameter Function (SDF) is a scalar function defined on a 3D mesh surface that measures the local volume thickness (diameter) around each point, providing a pose-oblivious, informative signature for shape analysis. It calculates an average ray intersection distance, heavily used in mesh segmentation and skeletonization to identify distinct, similar components.


In this project, we try to optimize SDF function by using CUDA C++, and compare it with the previous work: Parallelization of Shape Diameter Function Computation using
OpenCL
The link of this work: https://old.cescg.org/CESCG-2014/papers/Kamenicky-Parallelization_of_Shape_Diameter_Function_Computation_using_OpenCL.pdf


The Speed improvement: 

# Performance Benchmark: OpenCL vs. NVIDIA OptiX

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
