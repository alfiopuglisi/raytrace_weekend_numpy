# raytrace_weekend_numpy

This is Python/numpy implementation of Raytracing in One Weekend: https://github.com/RayTracing/raytracing.github.io

A straight Python translation would have resulted in prohibitive runtimes, since Python is a very high level language with poor maths performance. The numpy module implements fast linear algebra using C or Fortran routines, and allows a Python program to perform acceptably (or even, in some cases, faster than a badly written C routine), while keeping Python's readability.

This is not exactly one of those cases: porting the raytracing code to numpy required significant refactoring, because the original C++ code was following one ray's route at a time. With numpy, which is efficient when rather big arrays are used, the only practical way is to compute the same operation in parallel on all rays at the same time. This leads to significant overheads when some computations are later discarded, or when these arrays must be shuffled around for the next computation step.

The resulting program is somewhat longer and, at times, less easy to follow. The rendering speed on my PC is 3x slower than the C++ version, which is not a bad result given all the limitations of the language.

```
time python main.py
1000x666 pixels, 3 samples per pixel
...
Elapsed time: 64.653 s
```
```
time python main_optimized.py
1000x666 pixels, 3 samples per pixel
...
Elapsed time: 28.727 s
```

