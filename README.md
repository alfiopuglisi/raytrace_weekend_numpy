# raytrace_weekend_numpy

This is Python/numpy implementation of Raytracing in One Weekend: https://github.com/RayTracing/raytracing.github.io

The Jupyter notebook replicates the C++ tutorial step-by-step: https://github.com/alfiopuglisi/raytrace_weekend_numpy/blob/master/raytrace_weekend_numpy.ipynb

A straight Python translation would have resulted in prohibitive runtimes, since Python is a very high level language with poor maths performance. The numpy module implements fast linear algebra using C or Fortran routines, and allows a Python program to perform acceptably (or even, in some cases, faster than a badly written C routine), while keeping Python's readability.

This is not exactly one of those cases: porting the raytracing code to numpy required significant refactoring, because the original C++ code was following one ray's route at a time. With numpy, which is efficient when rather big arrays are used, the only practical way is to compute the same operation in parallel on all rays at the same time. This leads to significant overheads when some computations are later discarded, or when these arrays must be shuffled around for the next computation step.

The resulting program is somewhat longer and, at times, less easy to follow. The rendering speed on my PC is 2.5x slower than the C++ version, which is not a bad result given all the limitations of the language.

```
Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz
```

Optimized version:
```
$ python main_optimized.py
1200x675 pixels, 10 samples per pixel

Elapsed time: 110.440 s
```

Non-optimizied version:
```
$ python main.py
1200x675 pixels, 10 samples per pixel

Elapsed time: 261.434 s
```


C++ version from  https://github.com/RayTracing/raytracing.github.io:
```
$ g++ -o ray main.cc -O2 -I../common -std=c++11
$ time ./ray  > img.ppm

Scanlines remaining: 0   
Done.

real	0m48.474s
```


