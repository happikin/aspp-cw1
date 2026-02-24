I have no name!@kgpu-s2824305-epcc-m8jcm-4xf7z:/home/eidf018/eidf018/shared/s2824305-epcc-pvc/aspp-cw1$ ./build-dev/awave -shape 256,256,256
Initialising CPU simulation from parameters...
Grid shape: [256, 256, 256]
Grid spacing: 10 m
Time step: 0.002 s
Number of steps: 100
Output period: 10
Initial pressure field is pulse in middle of domain
Speed of sound is simple ocean depth model
Damping field to avoid reflections in x & y directions (large quiet ocean)
Write initial conditions to test.cpu.vtkhdf
Initialising CUDA simulation as copy of CPU...
Write initial conditions to test.cuda.vtkhdf
Initialising OpenMP simulation as copy of CPU...
Write initial conditions to test.omp.vtkhdf
Starting run with CPU, timing in 10 chunks
Chunk 0, length = 10, time = 0.61455935 s
Chunk 1, length = 10, time = 0.6192921 s
Chunk 2, length = 10, time = 0.6019939 s
Chunk 3, length = 10, time = 0.611999 s
Chunk 4, length = 10, time = 0.6083726 s
Chunk 5, length = 10, time = 0.6095211 s
Chunk 6, length = 10, time = 0.60744697 s
Chunk 7, length = 10, time = 0.60186994 s
Chunk 8, length = 10, time = 0.6034849 s
Chunk 9, length = 10, time = 0.59917223 s
Starting run with CUDA, timing in 10 chunks
Chunk 0, length = 10, time = 0.1408048 s
Chunk 1, length = 10, time = 0.1374926 s
Chunk 2, length = 10, time = 0.14566913 s
Chunk 3, length = 10, time = 0.1485015 s
Chunk 4, length = 10, time = 0.142628 s
Chunk 5, length = 10, time = 0.14484352 s
Chunk 6, length = 10, time = 0.143021 s
Chunk 7, length = 10, time = 0.14464265 s
Chunk 8, length = 10, time = 0.14949737 s
Chunk 9, length = 10, time = 0.14125787 s
Checking CUDA results...
Number of differences detected = 0
Starting run with OpenMP, timing in 10 chunks
Chunk 0, length = 10, time = 0.14502658 s
Chunk 1, length = 10, time = 0.1391716 s
Chunk 2, length = 10, time = 0.13153054 s
Chunk 3, length = 10, time = 0.12950848 s
Chunk 4, length = 10, time = 0.13204606 s
Chunk 5, length = 10, time = 0.13173711 s
Chunk 6, length = 10, time = 0.14244613 s
Chunk 7, length = 10, time = 0.13768737 s
Chunk 8, length = 10, time = 0.13279808 s
Chunk 9, length = 10, time = 0.12927894 s
Checking OpenMP results...
Number of differences detected = 0
Summary for CPU
Run time / seconds
min = 5.992e-01, max = 6.193e-01, mean = 6.078e-01, std = 6.331e-03
Performance / (site updates per second)
min = 2.709e+08, max = 2.800e+08, mean = 2.761e+08, std = 2.866e+06
Summary for CUDA
Run time / seconds
min = 1.375e-01, max = 1.495e-01, mean = 1.438e-01, std = 3.606e-03
Performance / (site updates per second)
min = 1.122e+09, max = 1.220e+09, mean = 1.167e+09, std = 2.932e+07
Summary for OpenMP
Run time / seconds
min = 1.293e-01, max = 1.450e-01, mean = 1.351e-01, std = 5.574e-03
Performance / (site updates per second)
min = 1.157e+09, max = 1.298e+09, mean = 1.243e+09, std = 5.010e+07
