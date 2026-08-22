# Optimizing Shape Diameter Function using High Performance Computing

![USTH](https://img.shields.io/badge/USTH-Final%20Project-blue)
![CUDA](https://img.shields.io/badge/CUDA-20.0-green)
![OptiX](https://img.shields.io/badge/OptiX-7.6-orange)

This project presents a **GPU-accelerated Shape Diameter Function (SDF) computation system** using NVIDIA OptiX. SDF assigns a scalar value to each vertex of a 3D mesh representing the local thickness of the model at that point — a fundamental geometric descriptor widely used in shape analysis, segmentation, and parameterization.

Unlike the traditional approach of casting rays on a CPU, our method uses **hardware-accelerated ray tracing (RT Cores)** to consistently outperform the PyMeshLab GPU reference implementation — from small 2k-vertex meshes up to million-vertex scans — while producing comparable output quality.

---

## What is the Shape Diameter Function?

Introduced by Shapira et al. [1], the SDF is a per-vertex scalar measure that captures the local "thickness" of a 3D model at any point on its surface. Conceptually, consider a point **p** on the mesh surface. If one looks along the inward normal direction $-\vec{\mathbf{n}}$, the SDF quantifies how far one must travel before exiting the opposite side of the object.

However, a single ray along the inward normal may yield unreliable results on bumpy or irregular interior surfaces. To address this, the SDF casts a cone of **R rays** (typically $R = 64$) from point **p**, directed toward the interior of the model. Each ray records its travel distance $d_i$ to the opposite wall, and the final SDF value is computed as a weighted average:

$$
\text{SDF}(p) = \frac{\sum (d_i \times w_i)}{\sum w_i}
$$

where $w_i = \frac{1}{\theta_i}$ is the weight inversely proportional to the angle between the ray and the cone axis.

<p align="center">
  <img src="image/112_optix.png" width="400" alt="SDF heatmap example">
  <br>
  <em>A 3D model with SDF values visualized as a heatmap. Red = thick, blue = thin.</em>
</p>

A thick region (like a human torso) produces large SDF values, while a thin region (like a finger) produces small values. Because the SDF depends purely on the object's volume, it remains **invariant to rigid transformations** — rotating or bending the model does not alter the thickness values.

---

## Why SDF: Role in 3D Object Analysis

Understanding an object's thickness provides essential geometric cues for:
- **Mesh segmentation**: Cutting a 3D model into logical pieces (e.g., separating an arm from a body because the wrist is thinner)
- **Skeletonization**: Finding the "bones" inside a 3D character by tracing the thickest parts
- **Shape matching and retrieval**: Identifying similar shapes across databases using thickness-based signatures that are pose-oblivious
- **Character animation**: Rigging and skinning by understanding volumetric properties

---

## Performance Benchmark: PyMeshLab GPU (VCGlib) vs. NVIDIA OptiX

| Model | Vertices | Faces | PyMeshLab GPU (s) | OptiX (s) |
| :--- | :---: | :---: | :---: | :---: |
| 125.obj | 1,343 | 2,682 | 0.6779 | 0.0046 |
| 398.obj | 1,480 | 2,956 | 0.5995 | 0.0194 |
| 353.obj | 1,512 | 3,020 | 0.5683 | 0.0047 |
| 200.obj | 1,515 | 3,026 | 0.7121 | 0.0034 |
| 52.obj | 1,554 | 3,104 | 0.5760 | 0.0032 |
| 341.obj | 1,663 | 3,322 | 0.5239 | 0.0052 |
| 342.obj | 1,663 | 3,322 | 0.6834 | 0.0048 |
| 343.obj | 1,663 | 3,322 | 0.6459 | 0.0048 |
| 58.obj | 1,857 | 3,710 | 0.5229 | 0.0058 |
| 47.obj | 2,028 | 4,052 | 0.4896 | 0.0036 |
| 386.obj | 2,087 | 4,170 | 0.5625 | 0.0053 |
| 257.obj | 2,135 | 4,266 | 0.5659 | 0.0066 |
| 360.obj | 2,200 | 4,396 | 0.6134 | 0.0076 |
| 60.obj | 2,375 | 4,746 | 0.4678 | 0.0048 |
| 46.obj | 2,394 | 4,784 | 0.4700 | 0.0031 |
| 185.obj | 2,489 | 4,974 | 0.7488 | 0.0054 |
| 53.obj | 2,496 | 4,988 | 0.5588 | 0.0039 |
| 260.obj | 2,497 | 4,990 | 1.1674 | 0.0057 |
| 348.obj | 2,568 | 5,132 | 0.5628 | 0.0052 |
| 349.obj | 2,568 | 5,132 | 0.5441 | 0.0051 |
| 347.obj | 2,623 | 5,242 | 0.6566 | 0.0083 |
| 9.obj | 2,639 | 5,274 | 0.6660 | 0.0106 |
| 55.obj | 2,850 | 5,696 | 0.5608 | 0.0066 |
| 51.obj | 2,858 | 5,712 | 0.6388 | 0.0059 |
| 59.obj | 2,895 | 5,786 | 0.5875 | 0.0051 |
| 354.obj | 2,910 | 5,816 | 0.5667 | 0.0103 |
| 124.obj | 3,101 | 6,198 | 1.0362 | 0.0080 |
| 49.obj | 3,110 | 6,220 | 0.5132 | 0.0101 |
| 243.obj | 3,158 | 6,312 | 0.6973 | 0.0066 |
| 251.obj | 3,158 | 6,312 | 0.6234 | 0.0097 |
| 344.obj | 3,182 | 6,360 | 0.5246 | 0.0071 |
| 43.obj | 3,414 | 6,824 | 0.4001 | 0.0119 |
| 241.obj | 3,478 | 6,952 | 0.6439 | 0.0100 |
| 346.obj | 3,587 | 7,170 | 0.6823 | 0.0097 |
| 345.obj | 3,614 | 7,224 | 0.6624 | 0.0108 |
| 254.obj | 3,638 | 7,272 | 0.6144 | 0.0067 |
| 400.obj | 3,703 | 7,402 | 0.5736 | 0.0064 |
| 249.obj | 3,714 | 7,424 | 0.5635 | 0.0114 |
| 214.obj | 3,804 | 7,604 | 0.5544 | 0.0058 |
| 367.obj | 3,900 | 7,800 | 0.5673 | 0.0082 |
| 203.obj | 3,906 | 7,808 | 0.6207 | 0.0049 |
| 399.obj | 3,911 | 7,818 | 0.5535 | 0.0079 |
| 219.obj | 3,942 | 7,880 | 0.5723 | 0.0073 |
| 213.obj | 3,963 | 7,922 | 0.7445 | 0.0111 |
| 42.obj | 4,164 | 8,324 | 0.5093 | 0.0144 |
| 246.obj | 4,219 | 8,434 | 0.6920 | 0.0125 |
| 250.obj | 4,280 | 8,556 | 0.5730 | 0.0128 |
| 211.obj | 4,284 | 8,564 | 0.6079 | 0.0096 |
| 212.obj | 4,298 | 8,592 | 0.5848 | 0.0065 |
| 391.obj | 4,315 | 8,626 | 0.5281 | 0.0137 |
| 204.obj | 4,407 | 8,810 | 0.5746 | 0.0067 |
| 237.obj | 4,439 | 8,874 | 0.6320 | 0.0110 |
| 220.obj | 4,457 | 8,910 | 0.6043 | 0.0141 |
| 215.obj | 4,463 | 8,922 | 0.6133 | 0.0105 |
| 205.obj | 4,478 | 8,952 | 0.5561 | 0.0061 |
| 216.obj | 4,478 | 8,952 | 0.5849 | 0.0110 |
| 201.obj | 4,487 | 8,970 | 0.6349 | 0.0066 |
| 202.obj | 4,491 | 8,978 | 0.6207 | 0.0062 |
| 244.obj | 4,501 | 8,998 | 0.6370 | 0.0140 |
| 184.obj | 4,685 | 9,366 | 0.7202 | 0.0130 |
| 1.obj | 4,706 | 9,408 | 1.5931 | 0.0286 |
| 384.obj | 4,712 | 9,420 | 0.7470 | 0.0124 |
| 206.obj | 4,794 | 9,584 | 0.6962 | 0.0080 |
| 236.obj | 4,845 | 9,686 | 0.6537 | 0.0091 |
| 74.obj | 5,044 | 10,084 | 0.6201 | 0.0116 |
| 255.obj | 5,054 | 10,104 | 0.6681 | 0.0112 |
| 224.obj | 5,061 | 10,118 | 0.6791 | 0.0125 |
| 238.obj | 5,096 | 10,188 | 0.5965 | 0.0109 |
| 73.obj | 5,109 | 10,214 | 0.6069 | 0.0114 |
| 209.obj | 5,110 | 10,216 | 0.6980 | 0.0108 |
| 239.obj | 5,121 | 10,238 | 0.5729 | 0.0095 |
| 48.obj | 5,176 | 10,348 | 0.5368 | 0.0122 |
| 351.obj | 5,192 | 10,380 | 0.4946 | 0.0139 |
| 309.obj | 5,197 | 10,390 | 0.8246 | 0.0099 |
| 217.obj | 5,201 | 10,398 | 0.5761 | 0.0116 |
| 207.obj | 5,202 | 10,400 | 0.6549 | 0.0083 |
| 226.obj | 5,216 | 10,428 | 0.5381 | 0.0098 |
| 227.obj | 5,216 | 10,428 | 0.5917 | 0.0107 |
| 72.obj | 5,228 | 10,452 | 0.5882 | 0.0114 |
| 229.obj | 5,245 | 10,486 | 0.6050 | 0.0283 |
| 235.obj | 5,250 | 10,496 | 0.6945 | 0.0103 |
| 222.obj | 5,255 | 10,506 | 0.7318 | 0.0084 |
| 45.obj | 5,399 | 10,794 | 0.5502 | 0.0146 |
| 61.obj | 5,400 | 10,796 | 0.6291 | 0.0112 |
| 232.obj | 5,440 | 10,876 | 0.5848 | 0.0153 |
| 210.obj | 5,508 | 11,012 | 0.7042 | 0.0277 |
| 247.obj | 5,519 | 11,034 | 0.5742 | 0.0132 |
| 63.obj | 5,519 | 11,034 | 0.5350 | 0.0133 |
| 396.obj | 5,538 | 11,072 | 0.6242 | 0.0161 |
| 68.obj | 5,583 | 11,162 | 0.6911 | 0.0096 |
| 228.obj | 5,592 | 11,180 | 0.6640 | 0.0153 |
| 12.obj | 5,614 | 11,224 | 0.5344 | 0.0092 |
| 62.obj | 5,619 | 11,234 | 0.5969 | 0.0182 |
| 15.obj | 5,631 | 11,258 | 0.7734 | 0.0110 |
| 70.obj | 5,631 | 11,258 | 0.5858 | 0.0121 |
| 3.obj | 5,641 | 11,278 | 0.6844 | 0.0120 |
| 4.obj | 5,676 | 11,348 | 0.5949 | 0.0152 |
| 14.obj | 5,691 | 11,378 | 0.7828 | 0.0123 |
| 114.obj | 5,794 | 11,588 | 0.4424 | 0.0142 |
| 71.obj | 5,849 | 11,694 | 0.5915 | 0.0095 |
| 92.obj | 5,867 | 11,730 | 0.8594 | 0.0112 |
| 76.obj | 5,923 | 11,842 | 0.5623 | 0.0098 |
| 121.obj | 5,944 | 11,888 | 0.6823 | 0.0168 |
| 256.obj | 5,970 | 11,936 | 0.5496 | 0.0130 |
| 225.obj | 6,076 | 12,148 | 0.7676 | 0.0085 |
| 208.obj | 6,104 | 12,204 | 0.6327 | 0.0119 |
| 129.obj | 6,144 | 12,284 | 0.6096 | 0.0118 |
| 98.obj | 6,154 | 12,304 | 0.8001 | 0.0164 |
| 128.obj | 6,164 | 12,324 | 0.8259 | 0.0131 |
| 253.obj | 6,164 | 12,324 | 0.5759 | 0.0098 |
| 218.obj | 6,198 | 12,392 | 0.6892 | 0.0091 |
| 231.obj | 6,264 | 12,524 | 0.5258 | 0.0229 |
| 37.obj | 6,270 | 12,540 | 0.8142 | 0.0133 |
| 44.obj | 6,288 | 12,572 | 0.7052 | 0.0130 |
| 126.obj | 6,325 | 12,646 | 0.7270 | 0.0135 |
| 252.obj | 6,351 | 12,698 | 0.5210 | 0.0112 |
| 81.obj | 6,370 | 12,736 | 0.7906 | 0.0106 |
| 83.obj | 6,376 | 12,748 | 0.6398 | 0.0156 |
| 65.obj | 6,448 | 12,892 | 0.4907 | 0.0131 |
| 245.obj | 6,475 | 12,946 | 0.6114 | 0.0149 |
| 77.obj | 6,587 | 13,170 | 0.5563 | 0.0146 |
| 186.obj | 6,607 | 13,210 | 0.6879 | 0.0148 |
| 223.obj | 6,656 | 13,308 | 0.8744 | 0.0155 |
| 372.obj | 6,684 | 13,368 | 0.5196 | 0.0121 |
| 378.obj | 6,684 | 13,368 | 0.5910 | 0.0158 |
| 79.obj | 6,689 | 13,374 | 0.5721 | 0.0196 |
| 69.obj | 6,701 | 13,398 | 0.6786 | 0.0141 |
| 64.obj | 6,797 | 13,590 | 0.5591 | 0.0136 |
| 240.obj | 6,833 | 13,662 | 0.6419 | 0.0147 |
| 84.obj | 6,850 | 13,696 | 0.6219 | 0.0154 |
| 99.obj | 6,881 | 13,758 | 0.7195 | 0.0160 |
| 379.obj | 6,923 | 13,842 | 0.5011 | 0.0155 |
| 381.obj | 6,938 | 13,872 | 0.5565 | 0.0136 |
| 192.obj | 6,991 | 13,978 | 0.5707 | 0.0177 |
| 41.obj | 7,016 | 14,028 | 0.7251 | 0.0074 |
| 86.obj | 7,038 | 14,072 | 0.6918 | 0.0116 |
| 196.obj | 7,112 | 14,220 | 0.5704 | 0.0174 |
| 221.obj | 7,121 | 14,238 | 0.7488 | 0.0165 |
| 94.obj | 7,236 | 14,468 | 0.5437 | 0.0177 |
| 181.obj | 7,242 | 14,480 | 0.6296 | 0.0194 |
| 122.obj | 7,251 | 14,498 | 0.6522 | 0.0158 |
| 397.obj | 7,268 | 14,532 | 0.6345 | 0.0625 |
| 193.obj | 7,314 | 14,624 | 0.6396 | 0.0185 |
| 33.obj | 7,349 | 14,698 | 0.5410 | 0.0170 |
| 80.obj | 7,351 | 14,698 | 0.6097 | 0.0177 |
| 95.obj | 7,355 | 14,706 | 0.5597 | 0.0188 |
| 54.obj | 7,407 | 14,810 | 0.6335 | 0.0074 |
| 57.obj | 7,413 | 14,822 | 0.5066 | 0.0086 |
| 50.obj | 7,420 | 14,836 | 0.6864 | 0.0109 |
| 78.obj | 7,470 | 14,936 | 0.6160 | 0.0177 |
| 87.obj | 7,470 | 14,936 | 0.6245 | 0.0119 |
| 234.obj | 7,498 | 14,992 | 0.5980 | 0.0156 |
| 190.obj | 7,500 | 14,996 | 0.8731 | 0.0177 |
| 197.obj | 7,553 | 15,102 | 0.8333 | 0.0221 |
| 230.obj | 7,573 | 15,142 | 0.7834 | 0.0127 |
| 187.obj | 7,628 | 15,256 | 0.7513 | 0.0151 |
| 67.obj | 7,651 | 15,298 | 0.5807 | 0.0181 |
| 89.obj | 7,654 | 15,304 | 0.6299 | 0.0216 |
| 66.obj | 7,739 | 15,474 | 0.5226 | 0.0173 |
| 130.obj | 7,747 | 15,490 | 0.5337 | 0.0129 |
| 132.obj | 7,812 | 15,620 | 0.6046 | 0.0108 |
| 242.obj | 7,849 | 15,694 | 0.9426 | 0.0181 |
| 90.obj | 7,877 | 15,750 | 0.6519 | 0.0181 |
| 139.obj | 8,017 | 16,030 | 0.8548 | 0.0546 |
| 111.obj | 8,050 | 16,100 | 0.8995 | 0.0180 |
| 100.obj | 8,064 | 16,124 | 0.6728 | 0.0058 |
| 382.obj | 8,078 | 16,152 | 0.5646 | 0.0207 |
| 97.obj | 8,167 | 16,330 | 0.6298 | 0.0389 |
| 88.obj | 8,210 | 16,416 | 0.5697 | 0.0256 |
| 93.obj | 8,216 | 16,432 | 0.6112 | 0.0273 |
| 310.obj | 8,263 | 16,522 | 0.5535 | 0.0224 |
| 82.obj | 8,299 | 16,594 | 0.6722 | 0.0235 |
| 85.obj | 8,388 | 16,772 | 0.5960 | 0.0239 |
| 388.obj | 8,411 | 16,818 | 0.4808 | 0.0234 |
| 7.obj | 8,441 | 16,878 | 0.6644 | 0.0219 |
| 110.obj | 8,456 | 16,920 | 0.9180 | 0.0299 |
| 194.obj | 8,471 | 16,938 | 0.6474 | 0.0272 |
| 101.obj | 8,499 | 16,998 | 0.7224 | 0.0055 |
| 96.obj | 8,504 | 17,004 | 0.6214 | 0.0256 |
| 195.obj | 8,524 | 17,044 | 0.5300 | 0.0243 |
| 188.obj | 8,603 | 17,202 | 0.6590 | 0.0295 |
| 392.obj | 8,618 | 17,232 | 0.5621 | 0.0246 |
| 91.obj | 8,640 | 17,276 | 0.6998 | 0.0256 |
| 199.obj | 8,647 | 17,290 | 0.7322 | 0.0492 |
| 104.obj | 8,653 | 17,306 | 0.9455 | 0.0124 |
| 75.obj | 8,679 | 17,354 | 0.6866 | 0.0244 |
| 350.obj | 8,759 | 17,514 | 0.4668 | 0.0477 |
| 56.obj | 8,771 | 17,538 | 0.8512 | 0.0145 |
| 258.obj | 8,946 | 17,888 | 1.1157 | 0.0239 |
| 394.obj | 9,009 | 18,014 | 0.6901 | 0.0270 |
| 259.obj | 9,015 | 18,026 | 0.7873 | 0.0251 |
| 36.obj | 9,076 | 18,152 | 0.5522 | 0.0352 |
| 137.obj | 9,124 | 18,244 | 0.8351 | 0.0293 |
| 118.obj | 9,153 | 18,306 | 0.5366 | 0.0315 |
| 140.obj | 9,200 | 18,396 | 0.6711 | 0.0254 |
| 390.obj | 9,239 | 18,474 | 0.4439 | 0.0305 |
| 301.obj | 9,252 | 18,500 | 0.5462 | 0.0268 |
| 105.obj | 9,261 | 18,530 | 0.7194 | 0.0084 |
| 143.obj | 9,270 | 18,536 | 0.6347 | 0.0271 |
| 380.obj | 9,356 | 18,708 | 0.6002 | 0.0282 |
| 175.obj | 9,480 | 18,956 | 0.7366 | 0.0104 |
| 389.obj | 9,490 | 18,976 | 0.4768 | 0.0300 |
| 10.obj | 9,508 | 19,012 | 0.5832 | 0.0053 |
| 180.obj | 9,548 | 19,092 | 0.8242 | 0.0350 |
| 198.obj | 9,564 | 19,124 | 0.7704 | 0.0361 |
| 138.obj | 9,594 | 19,184 | 0.6952 | 0.0655 |
| 34.obj | 9,602 | 19,204 | 0.5342 | 0.0296 |
| 103.obj | 9,652 | 19,304 | 0.7847 | 0.0053 |
| 393.obj | 9,757 | 19,510 | 0.5574 | 0.0308 |
| torus.obj | 9,801 | 19,602 | 0.5426 | 0.0284 |
| 142.obj | 9,802 | 19,600 | 0.6146 | 0.0262 |
| 115.obj | 10,043 | 20,086 | 0.4528 | 0.0296 |
| 2.obj | 10,050 | 20,096 | 0.6899 | 0.0228 |
| 160.obj | 10,082 | 20,160 | 0.9190 | 0.0303 |
| 162.obj | 10,096 | 20,188 | 0.6927 | 0.0328 |
| 6.obj | 10,098 | 20,192 | 0.7507 | 0.0324 |
| 156.obj | 10,100 | 20,196 | 1.1676 | 0.0388 |
| 120.obj | 10,121 | 20,242 | 0.6707 | 0.0347 |
| 172.obj | 10,141 | 20,278 | 0.9078 | 0.0180 |
| 233.obj | 10,186 | 20,368 | 0.6610 | 0.0550 |
| 165.obj | 10,233 | 20,462 | 0.8080 | 0.0320 |
| 182.obj | 10,283 | 20,562 | 0.8452 | 0.0349 |
| 109.obj | 10,301 | 20,602 | 0.7147 | 0.0199 |
| 352.obj | 10,400 | 20,796 | 0.5364 | 0.0357 |
| 385.obj | 10,436 | 20,868 | 0.4878 | 0.0299 |
| 133.obj | 10,466 | 20,932 | 0.7535 | 0.0202 |
| 108.obj | 10,500 | 21,008 | 0.9233 | 0.0056 |
| 152.obj | 10,543 | 21,082 | 0.6114 | 0.0390 |
| 373.obj | 10,556 | 21,112 | 0.5357 | 0.0301 |
| 366.obj | 10,637 | 21,274 | 0.5559 | 0.0347 |
| 113.obj | 10,748 | 21,500 | 0.6186 | 0.0340 |
| 168.obj | 10,752 | 21,500 | 0.8513 | 0.0334 |
| 313.obj | 10,852 | 21,700 | 0.8690 | 0.0183 |
| 163.obj | 10,879 | 21,758 | 0.8750 | 0.0380 |
| 134.obj | 10,932 | 21,860 | 0.7039 | 0.0465 |
| 11.obj | 10,999 | 21,994 | 0.8409 | 0.0206 |
| 8.obj | 11,015 | 22,026 | 0.6655 | 0.0492 |
| 131.obj | 11,051 | 22,098 | 0.5676 | 0.0193 |
| 166.obj | 11,090 | 22,176 | 0.8082 | 0.0334 |
| 368.obj | 11,202 | 22,400 | 0.5933 | 0.0407 |
| 395.obj | 11,312 | 22,620 | 0.7613 | 0.0462 |
| 183.obj | 11,413 | 22,822 | 0.7154 | 0.0386 |
| 107.obj | 11,421 | 22,854 | 0.8622 | 0.0076 |
| 337.obj | 11,533 | 23,062 | 0.5102 | 0.0463 |
| 375.obj | 11,762 | 23,520 | 0.6259 | 0.0352 |
| 248.obj | 11,790 | 23,576 | 0.7867 | 0.0362 |
| 127.obj | 11,906 | 23,808 | 0.7508 | 0.0247 |
| 178.obj | 12,175 | 24,346 | 0.7220 | 0.0197 |
| 119.obj | 12,326 | 24,652 | 0.6110 | 0.0390 |
| 170.obj | 12,500 | 24,996 | 0.7946 | 0.0427 |
| 176.obj | 12,500 | 24,996 | 0.8434 | 0.0061 |
| 177.obj | 12,543 | 25,082 | 0.7481 | 0.0072 |
| 169.obj | 12,561 | 25,118 | 0.7488 | 0.0407 |
| 376.obj | 12,568 | 25,132 | 0.5602 | 0.0569 |
| 167.obj | 12,647 | 25,290 | 0.8280 | 0.0458 |
| 151.obj | 12,696 | 25,388 | 0.6380 | 0.0593 |
| 174.obj | 12,831 | 25,658 | 0.6801 | 0.0090 |
| 145.obj | 13,206 | 26,408 | 0.6311 | 0.0426 |
| 179.obj | 13,324 | 26,644 | 0.6622 | 0.0205 |
| 383.obj | 13,331 | 26,658 | 0.7456 | 0.0450 |
| 148.obj | 13,375 | 26,746 | 0.6757 | 0.0454 |
| 363.obj | 13,434 | 26,864 | 0.6285 | 0.0483 |
| 116.obj | 13,463 | 26,926 | 0.4611 | 0.0590 |
| 365.obj | 13,514 | 27,028 | 0.5428 | 0.0489 |
| 364.obj | 13,548 | 27,100 | 0.6332 | 0.0449 |
| 150.obj | 13,579 | 27,154 | 0.6206 | 0.0521 |
| 153.obj | 13,581 | 27,158 | 0.6558 | 0.0651 |
| 149.obj | 13,588 | 27,172 | 0.5575 | 0.0496 |
| 369.obj | 13,606 | 27,212 | 0.5912 | 0.0470 |
| 112.obj | 13,628 | 27,256 | 1.4853 | 0.0465 |
| 13.obj | 13,703 | 27,402 | 0.7312 | 0.0896 |
| 147.obj | 13,705 | 27,406 | 0.6076 | 0.0434 |
| 146.obj | 13,714 | 27,424 | 0.7046 | 0.0458 |
| 191.obj | 13,766 | 27,528 | 0.9857 | 0.0475 |
| 161.obj | 13,826 | 27,648 | 0.9454 | 0.0471 |
| 173.obj | 13,867 | 27,730 | 0.6470 | 0.0072 |
| 157.obj | 13,883 | 27,762 | 0.7308 | 0.1100 |
| 370.obj | 13,920 | 27,836 | 0.5532 | 0.0844 |
| 141.obj | 13,926 | 27,848 | 0.7199 | 0.0563 |
| 154.obj | 13,929 | 27,854 | 0.7572 | 0.0688 |
| 159.obj | 13,930 | 27,856 | 0.6137 | 0.0217 |
| 106.obj | 14,052 | 28,104 | 0.7298 | 0.0068 |
| 189.obj | 14,084 | 28,164 | 0.7421 | 0.0448 |
| 136.obj | 14,126 | 28,248 | 0.7825 | 0.0493 |
| 358.obj | 14,127 | 28,266 | 0.9047 | 0.0544 |
| 117.obj | 14,372 | 28,744 | 0.6782 | 0.0706 |
| 155.obj | 14,384 | 28,764 | 1.3135 | 0.0708 |
| 362.obj | 14,476 | 28,952 | 0.6742 | 0.0495 |
| 164.obj | 14,509 | 29,014 | 0.6395 | 0.0543 |
| 158.obj | 14,587 | 29,170 | 0.6146 | 0.0198 |
| 371.obj | 14,599 | 29,194 | 0.5464 | 0.0467 |
| 387.obj | 14,680 | 29,356 | 0.4841 | 0.0527 |
| 32.obj | 14,751 | 29,502 | 0.5702 | 0.0232 |
| 324.obj | 14,812 | 29,620 | 0.5863 | 0.0586 |
| 361.obj | 14,859 | 29,734 | 0.5717 | 0.0632 |
| 357.obj | 14,872 | 29,764 | 0.8655 | 0.0519 |
| 374.obj | 14,872 | 29,744 | 0.5621 | 0.0569 |
| 171.obj | 14,905 | 29,806 | 0.8575 | 0.1575 |
| 327.obj | 14,937 | 29,870 | 0.6131 | 0.0538 |
| 333.obj | 14,942 | 29,880 | 0.5211 | 0.0520 |
| 359.obj | 14,956 | 29,924 | 0.5611 | 0.0529 |
| 322.obj | 14,991 | 29,978 | 0.5685 | 0.0499 |
| 323.obj | 14,991 | 29,978 | 0.7040 | 0.0646 |
| 356.obj | 14,992 | 29,980 | 0.4560 | 0.0624 |
| 328.obj | 14,994 | 29,984 | 0.6598 | 0.0797 |
| 355.obj | 14,994 | 29,984 | 0.4659 | 0.0560 |
| 321.obj | 14,995 | 29,986 | 0.5471 | 0.0538 |
| 326.obj | 14,995 | 29,986 | 0.5208 | 0.0490 |
| 335.obj | 14,995 | 29,986 | 0.5254 | 0.0545 |
| 332.obj | 14,997 | 29,990 | 0.5177 | 0.0453 |
| 334.obj | 14,997 | 29,990 | 0.5679 | 0.0562 |
| 340.obj | 14,997 | 29,990 | 0.5323 | 0.0574 |
| 325.obj | 14,999 | 29,994 | 0.5727 | 0.0547 |
| 329.obj | 14,999 | 29,994 | 0.6454 | 0.0702 |
| 293.obj | 15,000 | 29,996 | 0.6929 | 0.0563 |
| 298.obj | 15,000 | 29,996 | 0.5639 | 0.0563 |
| 330.obj | 15,000 | 29,996 | 0.5569 | 0.0557 |
| 331.obj | 15,000 | 29,996 | 0.5842 | 0.0491 |
| 339.obj | 15,000 | 29,996 | 0.5851 | 0.0494 |
| 22.obj | 15,002 | 30,004 | 0.6833 | 0.0520 |
| 30.obj | 15,006 | 30,008 | 0.4160 | 0.0512 |
| 23.obj | 15,037 | 30,074 | 0.7351 | 0.1151 |
| 38.obj | 15,064 | 30,128 | 0.5969 | 0.0523 |
| 28.obj | 15,070 | 30,140 | 0.6373 | 0.0631 |
| 144.obj | 15,082 | 30,160 | 0.6158 | 0.0451 |
| 25.obj | 15,087 | 30,170 | 0.5571 | 0.0538 |
| 29.obj | 15,127 | 30,254 | 0.6501 | 0.0246 |
| 35.obj | 15,136 | 30,268 | 0.5497 | 0.0534 |
| 27.obj | 15,137 | 30,274 | 0.8844 | 0.0672 |
| 5.obj | 15,154 | 30,308 | 0.6700 | 0.1193 |
| 39.obj | 15,161 | 30,322 | 0.5561 | 0.0550 |
| 40.obj | 15,165 | 30,326 | 0.5674 | 0.0761 |
| 377.obj | 15,169 | 30,342 | 0.7480 | 0.0556 |
| 21.obj | 15,198 | 30,396 | 0.7775 | 0.0561 |
| 135.obj | 15,201 | 30,398 | 0.7549 | 0.0695 |
| 26.obj | 15,209 | 30,418 | 1.1395 | 0.0546 |
| 16.obj | 15,223 | 30,450 | 1.0286 | 0.0220 |
| 31.obj | 15,227 | 30,454 | 0.6691 | 0.0563 |
| 24.obj | 15,246 | 30,492 | 0.8771 | 0.0545 |
| 297.obj | 15,315 | 30,626 | 0.7363 | 0.0602 |
| 18.obj | 15,385 | 30,766 | 0.9446 | 0.0325 |
| 19.obj | 15,477 | 30,950 | 0.6833 | 0.0632 |
| 123.obj | 15,493 | 30,982 | 0.8323 | 0.0636 |
| 336.obj | 15,505 | 31,006 | 0.6519 | 0.0574 |
| 303.obj | 15,516 | 31,028 | 0.7822 | 0.0573 |
| 20.obj | 15,700 | 31,396 | 0.7472 | 0.0236 |
| 102.obj | 15,724 | 31,456 | 0.9287 | 0.0103 |
| 296.obj | 15,832 | 31,660 | 0.6993 | 0.0638 |
| 17.obj | 15,910 | 31,816 | 0.7538 | 0.0569 |
| 294.obj | 16,501 | 32,998 | 0.7863 | 0.0565 |
| 295.obj | 16,780 | 33,556 | 0.6681 | 0.0655 |
| 299.obj | 17,117 | 34,230 | 0.7495 | 0.0705 |
| 300.obj | 17,453 | 34,902 | 0.4712 | 0.0618 |
| highpeeling_coil.obj | 18,000 | 35,976 | 1.2812 | 0.0880 |
| 338.obj | 19,403 | 38,802 | 0.6466 | 0.0716 |
| 288.obj | 20,000 | 39,996 | 1.0080 | 0.0827 |
| 292.obj | 21,774 | 43,544 | 0.7665 | 0.0960 |
| 316.obj | 23,395 | 46,786 | 0.7616 | 0.0892 |
| 308.obj | 23,976 | 47,948 | 0.6497 | 0.1113 |
| 282.obj | 24,473 | 48,942 | 0.8845 | 0.1129 |
| 283.obj | 24,484 | 48,964 | 1.1166 | 0.0418 |
| 291.obj | 24,615 | 49,226 | 0.8843 | 0.0446 |
| leaf.obj | 24,866 | 49,728 | 0.5327 | 0.1043 |
| 286.obj | 25,108 | 50,212 | 0.9802 | 0.0462 |
| 302.obj | 25,125 | 50,246 | 0.8751 | 0.1178 |
| 318.obj | 25,145 | 50,286 | 0.6411 | 0.1107 |
| 285.obj | 25,193 | 50,382 | 0.9678 | 0.0564 |
| 289.obj | 25,211 | 50,418 | 0.8498 | 0.0450 |
| 311.obj | 25,230 | 50,456 | 0.8160 | 0.1248 |
| 281.obj | 25,273 | 50,542 | 0.7303 | 0.1198 |
| 290.obj | 25,431 | 50,858 | 0.7894 | 0.1754 |
| 304.obj | 25,467 | 50,930 | 0.7129 | 0.1113 |
| 287.obj | 25,491 | 50,978 | 1.1486 | 0.1311 |
| 284.obj | 25,494 | 50,984 | 1.0624 | 0.0550 |
| 315.obj | 25,768 | 51,532 | 0.6683 | 0.0574 |
| 314.obj | 26,437 | 52,870 | 0.5885 | 0.1202 |
| 319.obj | 26,558 | 53,112 | 0.6157 | 0.0468 |
| 306.obj | 26,798 | 53,592 | 0.6828 | 0.0439 |
| 312.obj | 26,985 | 53,966 | 0.6691 | 0.0596 |
| 320.obj | 27,118 | 54,232 | 0.6065 | 0.1378 |
| 307.obj | 27,439 | 54,874 | 0.7081 | 0.1476 |
| 317.obj | 27,726 | 55,448 | 0.6726 | 0.0482 |
| 305.obj | 27,824 | 55,644 | 0.7868 | 0.1564 |
| lucy.obj | 49,987 | 99,970 | 0.8514 | 0.1936 |
| armadillo.obj | 172,974 | 345,944 | 1.1379 | 0.2489 |
| dragon.obj | 437,645 | 871,414 | 1.6002 | 0.4647 |
| nefertiti.obj | 1,009,118 | 2,018,232 | 2.0473 | 0.8363 |

**Key findings:**
- **OptiX is faster across all 387 models**, spanning meshes from ~2k to over 1M vertices (Princeton Segmentation Benchmark + large scans)
- **PyMeshLab never completes in less than ~0.4 seconds**, regardless of model size, due to the fixed overhead of its camera initialization and OpenGL context setup
- **OptiX computes SDF for the million-vertex Nefertiti scan in 0.84 seconds**, making it suitable for interactive applications even on massive meshes

### Test Settings

| Parameter | OptiX (CUDA) | PyMeshLab GPU (VCGlib) |
| :--- | :--- | :--- |
| **Rays per vertex** | 64 | 64 |
| **Cone angle** | 150 degrees | 150 degrees |
| **Ray sampling** | Hammersley 2D (quasi-random) | Fibonacci sphere (uniform) |
| **Depth peeling layers** | N/A (single closest-hit via hardware BVH) | 10 layers |
| **Post-processing** | Log normalization + 3x anisotropic bilateral smoothing | None |
| **GPU** | NVIDIA RTX (OptiX RT Cores) | Any GPU with OpenGL 3.3+ |

---

## Methodology & Algorithmic Foundations

The Shape Diameter Function (SDF) pipeline transforms raw 3D polygon meshes into smooth, surface-aligned volumetric thickness values using NVIDIA OptiX. As detailed in Chapter 3 of the project report, the overall workflow is organized into four main phases: **Read OBJ Model**, **Parallel GPU Initialization**, **OptiX Ray Tracing Engine**, and **GPU Post-Processing**.

```mermaid
flowchart TD
    subgraph Phase1 ["Phase 1: CPU Reading"]
        A["Read OBJ Model (Model.cu)"]
    end

    subgraph Phase2 ["Phase 2: Parallel GPU Initialization (CUDA Streams)"]
        A --> B1["Calculate Vertex Normals (ModelHelper.cu)<br/>- Area-Weighted Cross Product<br/>- atomicAdd Accumulation<br/>- Unit Normalization"]
        A --> B2["Build BVH (OptixRunner.cuh)<br/>- optixAccelBuild<br/>- Disable Triangle Face Culling"]
    end

    subgraph Phase3 ["Phase 3: OptiX Ray Tracing Engine"]
        B1 --> C1["Build Shader Binding Table (OptixRunner.cuh)"]
        B2 --> C1
        C1 --> C2["OptiX Pipeline Setup & Cone Raygen (SDFOptix.cu)<br/>- 64 Rays/Vertex, 150° Cone<br/>- Hammersley 2D Sampling<br/>- Frisvad Local Tangent Frame<br/>- Hardware RT Core Traversal"]
        C2 --> C3["Calculate Weighted Raw SDF (SDFKernels.cuh)<br/>- Angle-Weighted Average"]
    end

    subgraph Phase4 ["Phase 4: GPU Post-Processing"]
        C3 --> D1["Normalize SDF (streamNorm)<br/>- Min-Max Scaling<br/>- Log Compression (alpha = 4.0)"]
        C3 --> D2["Build Adjacency Graph (streamCSR)<br/>- Generate Directed Edges<br/>- CUB RadixSort & Unique<br/>- CSR Format (row_ptr, col_ind)"]
        D1 --> E["3x Anisotropic Bilateral Smoothing (SDFKernels.cuh)<br/>- Ping-Pong Double Buffering<br/>- Spatial & Range Gaussians"]
        D2 --> E
    end

    E --> F["Final Per-Vertex SDF Map"]
```

---

### Phase 1: Read OBJ Model

The pipeline begins by loading raw 3D mesh data from Wavefront `.obj` files ([`Model.cu`](file:///e:/Code/FinalProject/Core/Model.cu)). A 3D surface is defined by vertex coordinates in 3D space (`v` lines) and triangular faces connecting triplets of vertices (`f` lines).

- **Index Conversion**: OBJ files use 1-based indexing. The parser converts indices to 0-based to align with GPU memory indexing.
- **Matrix Upload**: Vertex positions and face connectivity indices are packed into host matrices and uploaded to GPU memory via `CopyToDevice()`:

$$
\mathbf{V} = \begin{bmatrix} x_0 & y_0 & z_0 \\ x_1 & y_1 & z_1 \\ \vdots & \vdots & \vdots \end{bmatrix}_{|V| \times 3}, \qquad \mathbf{F} = \begin{bmatrix} v_0^0 & v_1^0 & v_2^0 \\ v_0^1 & v_1^1 & v_2^1 \\ \vdots & \vdots & \vdots \end{bmatrix}_{|F| \times 3}
$$

---

### Phase 2: Parallel GPU Initialization

To minimize pipeline startup overhead, two heavy initialization tasks are executed concurrently on independent CUDA streams (`streamNorm` and `streamBVH`):

#### 2.1 Calculate Vertex Normals ([`ModelHelper.cu`](file:///e:/Code/FinalProject/Core/ModelHelper.cu))

Vertex normals define inward directions for interior ray tracing. They are computed in two parallel GPU kernel passes:

##### Step 1: Area-Weighted Face Normal Accumulation (`GPUNormalCaculation`)

For a triangle with vertices $\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2$, the unnormalized face normal $\vec{n}_f$ is the cross product of its edge vectors $\vec{e}_1 = \mathbf{v}_1 - \mathbf{v}_0$ and $\vec{e}_2 = \mathbf{v}_2 - \mathbf{v}_0$:

$$
\vec{n}_f = \vec{e}_1 \times \vec{e}_2
$$

The magnitude $|\vec{n}_f|$ equals twice the triangle area, naturally weighting larger faces heavier. CUDA atomic addition (`atomicAdd`) safely accumulates face normals into adjacent vertices across concurrent threads:

$$
\vec{N}_{v_j} = \sum_{f \in \mathcal{F}(v_j)} \vec{n}_f
$$

##### Step 2: Unit Normalization (`GPUNormalizeVertexNormal`)

Accumulated vectors are normalized to unit length:

$$
\hat{\mathbf{n}}_v = \begin{cases} \frac{\vec{N}_v}{|\vec{N}_v|} & \text{if } |\vec{N}_v| > 0 \\ (0, 0, 1)^T & \text{otherwise} \end{cases}
$$

#### 2.2 Build BVH Acceleration Structure ([`OptixRunner.cuh`](file:///e:/Code/FinalProject/src/Optix/OptixRunner.cuh))

To enable real-time ray traversal across millions of triangles, an OptiX Bounding Volume Hierarchy (BVH) Geometry Acceleration Structure (GAS) is built via `optixAccelBuild`:
- **`OPTIX_BUILD_FLAG_PREFER_FAST_TRACE`**: Optimizes BVH tree node layout for maximum RT Core ray traversal speed.
- **`OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING`**: Crucial switch enabling double-sided ray-triangle intersections, allowing inward rays to strike interior back-walls.

---

### Phase 3: OptiX Ray Tracing Engine

#### 3.1 Build Shader Binding Table (SBT)

The Shader Binding Table acts as a fast execution directory mapping ray tracing events to GPU device programs ([`OptixRunner.cuh`](file:///e:/Code/FinalProject/src/Optix/OptixRunner.cuh)):
- **Raygen Record**: Pointing to `__raygen__sdf_cone` for shooting cone rays.
- **Miss Record**: Null program, ignoring rays that miss geometry.
- **Hitgroup Record**: Pointing to `__closesthit__sdf` for recording travel distances.

#### 3.2 OptiX Pipeline Setup & Ray Generation ([`SDFOptix.cu`](file:///e:/Code/FinalProject/src/Optix/SDFOptix.cu))

The OptiX pipeline executes the core physics simulation by launching 64 rays per vertex into the mesh interior:

##### 1. Local Tangent Frame
An orthonormal coordinate basis $(\mathbf{T}, \mathbf{B}, -\hat{\mathbf{n}}_v)$ is generated around the inward normal using the Frisvad method.

##### 2. Hammersley 2D Low-Discrepancy Sampling
Ray directions inside a cone aperture of $\theta_{\text{max}} = 150^\circ$ ($2.61799\text{ rad}$) are generated using the base-2 Van der Corput sequence $\Phi_2(i)$:

$$
x_i = \frac{i}{R}, \quad y_i = \Phi_2(i) = \sum_{k=0}^{\lfloor \log_2 i \rfloor} b_k \cdot 2^{-(k+1)}, \quad \text{for } i \in \{0, \dots, 63\}
$$

$$
\theta_i = x_i \cdot \frac{\theta_{\text{max}}}{2}, \quad \phi_i = 2\pi y_i
$$

$$
\mathbf{d}_{\text{world}} = (\sin\theta_i \cos\phi_i) \mathbf{T} + (\sin\theta_i \sin\phi_i) \mathbf{B} + (\cos\theta_i) (-\hat{\mathbf{n}}_v)
$$

##### 3. Hardware BVH Traversal & Payload
`optixTrace` queries RT Cores. The closest-hit program `__closesthit__sdf` records distance $d_i = \text{optixGetRayTmax()}$ into a 32-bit payload. Any-Hit programs are disabled (`OPTIX_RAY_FLAG_DISABLE_ANYHIT`) for peak hardware throughput.

#### 3.3 Calculate Weighted Raw SDF ([`SDFKernels.cuh`](file:///e:/Code/FinalProject/src/Optix/SDFKernels.cuh))

The `GPUComputeRawSDF` kernel aggregates valid ray travel distances using an angle-weighted average, where weight $w_i = \frac{1}{\theta_i + \epsilon}$ penalizes wide-angle deflections:

$$
\text{SDF}_{\text{raw}}(p) = \frac{\sum_{i=1}^{R_{\text{hit}}} d_i \cdot w_i}{\sum_{i=1}^{R_{\text{hit}}} w_i}
$$

---

### Phase 4: GPU Post-Processing

To transform raw distances into a smooth heat map, the pipeline runs normalization and adjacency graph construction in parallel on separate CUDA streams before a final bilateral smoothing pass.

#### Step-by-Step SDF Post-Processing Pipeline

| (a) Raw SDF | (b) Normalized SDF | (c) Smoothed SDF |
| :---: | :---: | :---: |
| <img src="image/no_normalization.png" width="100%"> | <img src="image/normalization.png" width="100%"> | <img src="image/smoothed.png" width="100%"> |
| *Raw unscaled distance values directly from ray tracing.* | *Logarithmic min-max scaling mapping values to $[0, 1]$.* | *3x anisotropic bilateral filtering preserving sharp features.* |

#### 4.1 Normalize SDF ([`SDFKernels.cuh`](file:///e:/Code/FinalProject/src/Optix/SDFKernels.cuh))

##### 1. Min-Max Scaling
`GPUComputeSDFMinMax` finds minimum ($\text{SDF}_{\min}$) and maximum ($\text{SDF}_{\max}$) values to scale raw distances to $[0, 1]$:

$$
\hat{v} = \frac{\text{SDF}_{\text{raw}}(p) - \text{SDF}_{\min}}{\text{SDF}_{\max} - \text{SDF}_{\min}}
$$

##### 2. Logarithmic Compression
`GPUApplySDFNormalization` compresses dynamic range with parameter $\alpha = 4.0$ to preserve fine details on thin geometry:

$$
v_{\text{norm}} = \frac{\ln(4.0 \cdot \hat{v} + 1.0)}{\ln(5.0)}
$$

#### 4.2 Build Adjacency Graph in CSR Format ([`SDFKernels.cuh`](file:///e:/Code/FinalProject/src/Optix/SDFKernels.cuh))

To enable 1-ring neighbor lookups without $O(|V|^2)$ memory overhead, mesh topology is packed into Compressed Sparse Row (CSR) format:

1. `GPUGenerateEdges`: Emits 6 directed edges per triangle face.
2. `cub::DeviceRadixSort::SortPairs`: Sorts edges by primary vertex ID on GPU.
3. `cub::DeviceSelect::Unique`: Removes duplicate manifold edges.
4. `GPUExtractCSR`: Converts unique edges into CSR arrays (`row_ptr` offset array `d_nbrOffsets` and `col_ind` list array `d_nbrLists`).

#### 4.3 Anisotropic Bilateral Smoothing ([`SDFKernels.cuh`](file:///e:/Code/FinalProject/src/Optix/SDFKernels.cuh))

The `AnisotropicSmoothingKernel` runs 3 iterations of bilateral filtering over the CSR 1-ring neighbor graph $\mathcal{N}(p)$ to eliminate high-frequency ray noise while preserving sharp feature boundaries:

$$
v_p^{(k+1)} = \frac{\sum_{q \in \mathcal{N}(p)} \exp\left(-\frac{\|\mathbf{x}_p - \mathbf{x}_q\|^2}{2\sigma_s^2}\right) \exp\left(-\frac{(v_p^{(k)} - v_q^{(k)})^2}{2\sigma_r^2}\right) v_q^{(k)}}{\sum_{q \in \mathcal{N}(p)} \exp\left(-\frac{\|\mathbf{x}_p - \mathbf{x}_q\|^2}{2\sigma_s^2}\right) \exp\left(-\frac{(v_p^{(k)} - v_q^{(k)})^2}{2\sigma_r^2}\right)}
$$

- **Spatial Gaussian**: $\sigma_s = 0.02 \cdot \text{diag}(\text{BoundingBox})$.
- **Range Gaussian**: $\sigma_r = 0.1$.
- **Ping-Pong Double Buffering**: Swaps input/output pointers (`d_sdfBuf1` $\leftrightarrow$ `d_sdfBuf2`) across iterations to prevent GPU memory race conditions.

---

## Quality Comparison

The following figures show the SDF heat map from both implementations and their absolute difference. Hot colors (red/yellow) indicate larger SDF values (thicker regions), while cool colors (blue) indicate thinner regions.

| Model | OptiX | PyMeshLab | Difference |
| :--- | :---: | :---: | :---: |
| **112** | <img src="image/112_optix.png" width="100%"> | <img src="image/112_pymeshlab.png" width="100%"> | <img src="image/112_diff.png" width="100%"> |
| **118** | <img src="image/118_optix.png" width="100%"> | <img src="image/118_pymeshlab.png" width="100%"> | <img src="image/118_diff.png" width="100%"> |
| **158** | <img src="image/158_optix.png" width="100%"> | <img src="image/158_pymeshlab.png" width="100%"> | <img src="image/158_diff.png" width="100%"> |
| **181** | <img src="image/181_optix.png" width="100%"> | <img src="image/181_pymeshlab.png" width="100%"> | <img src="image/181_diff.png" width="100%"> |
| **368** | <img src="image/368_optix.png" width="100%"> | <img src="image/368_pymeshlab.png" width="100%"> | <img src="image/368_diff.png" width="100%"> |
| **369** | <img src="image/369_optix.png" width="100%"> | <img src="image/369_pymeshlab.png" width="100%"> | <img src="image/369_diff.png" width="100%"> |
| **371** | <img src="image/371_optix.png" width="100%"> | <img src="image/371_pymeshlab.png" width="100%"> | <img src="image/371_diff.png" width="100%"> |
| **400** | <img src="image/400_optix.png" width="100%"> | <img src="image/400_pymeshlab.png" width="100%"> | <img src="image/400_diff.png" width="100%"> |
| **76** | <img src="image/76_optix.png" width="100%"> | <img src="image/76_pymeshlab.png" width="100%"> | <img src="image/76_diff.png" width="100%"> |
| **9** | <img src="image/9_optix.png" width="100%"> | <img src="image/9_pymeshlab.png" width="100%"> | <img src="image/9_diff.png" width="100%"> |

The OptiX bilateral filter produces smoother results in uniform thickness regions, while the PyMeshLab depth peeling approach exhibits high-frequency noise on constant-thickness surfaces. The overall thickness distribution captured by both methods is consistent.

---

## Limitations

- **Static meshes only**: The system currently does not handle animated or deforming geometry, as the BVH is built once and not updated during animation.
- **Single reference comparison**: Evaluation is limited to PyMeshLab; a broader comparison against other SDF methods would strengthen the assessment.

---

## Setup and Run

> **NOTE: Windows Only**
> This project is designed for **Windows environments** and will not compile or run correctly on Linux or macOS.

### Prerequisites

- **Visual Studio**: C++ toolchain (MSVC)
- **nvcc**: NVIDIA CUDA Compiler toolkit (CUDA 20.0+)
- **NVIDIA RTX GPU**: For hardware-accelerated ray tracing (RT Cores)
- **OptiX 7.6 SDK**: Available from NVIDIA Developer
- **CMake**: Build system generator

### Build

```bat
git clone https://github.com/TommyDatLC/OptimizeSDF
cd OptimizeSDF
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Run

```bat
OptimizeSDF.exe
```

The program will iterate over all `.obj` models in the `Model/` directory, compute SDF using the OptiX pipeline, and compare against the PyMeshLab reference.

### Source Files

| File | Role |
|------|------|
| `main.cu` | Entry point: orchestrates full pipeline |
| `Core/Model.cu` | 3D mesh model reader (OBJ parser) |
| `Core/ModelHelper.cu` | GPU kernels for vertex normal computation |
| `src/Optix/SDFOptix.cu` | OptiX shaders: raygen + closest-hit (compiled to PTX) |
| `src/Optix/SDFKernels.cuh` | CUDA kernels: raw SDF, normalization, CSR graph, bilateral smoothing |
| `src/Optix/OptixRunner.cuh` | OptiX pipeline setup, BVH build, SBT construction |
| `src/Optix/interface.cu` | High-level orchestration of the full SDF pipeline |

---

## References

1. L. Shapira, A. Shamir, D. Cohen-Or, "Consistent Mesh Partitioning and Skeletonization using the Shape Diameter Function," *Visual Computer*, vol. 24, pp. 249–259, 2008.
2. J. Kamenický, "Parallelization of Shape Diameter Function Computation using OpenCL," *Proceedings of CESCG*, 204, 2014.
3. S. Chen, T. Liu, Z. Shu, S. Xin, Y. He, C. Tu, "Fast and robust shape diameter function," *PLoS ONE*, vol. 13, no. 1, e0190666, 2018.
4. P. Cignoni et al., "MeshLab: an Open-Source Mesh Processing Tool," *Sixth Eurographics Italian Chapter Conference*, pp. 129–136, 2008.
5. S. G. Parker et al., "OptiX: a general purpose ray tracing engine," *ACM SIGGRAPH 2010 papers*, pp. 1–13, 2010.
6. J. Nickolls et al., "Scalable parallel programming with CUDA," *Queue*, vol. 6, no. 2, pp. 40–53, 2008.
7. C. Tomasi and R. Manduchi, "Bilateral Filtering for Gray and Color Images," *ICCV*, pp. 839–846, 1998.
8. NVIDIA Corporation, "CUB: Cooperative primitives for CUDA C++."
9. X. Chen, A. Golovinskiy, T. Funkhouser, "A Benchmark for 3D Mesh Segmentation," *ACM SIGGRAPH*, 2009.
10. A. Baldacci et al., "GPU-based approaches for shape diameter function computation and its applications focused on skeleton extraction," *Computers & Graphics*, vol. 59, pp. 151–159, 2016.
