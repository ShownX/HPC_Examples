# HPC_Examples
High Performance Computing Examples written for the class.

## matrixMultiplication.c
Matrix and matrix multiplication using OpenMP.
### Introduction
1. matrixMultiplication_noBlock(int matrix_size, int nthreads): matrix mutiplication without block;
2. matrixMultiplication_Block(int matrix_size, int block_size, int nthreads): matrix mutiplication using block;
3. matrixMultiplication_Recursive(int matrix_size, int block_size, int nthreads, int flag): matrix mutiplication recursively, an implementation of Strassen algorithm;
4. matrixMultiplication_mkl(int matrix_size, int nthreads): matrix mutiplication using mkl.

### Library
1. [papi](http://icl.cs.utk.edu/papi/): to measure the performance of the matrix multiplication.
2. [mkl](https://software.intel.com/en-us/intel-mkl): to measure as an ambition to catch.

### Build
Due to using the MKL library as comparision in the code, so we only can use intel compiler to build the code. Otherwise, you should delete the header *"mkl.h"* and comment the function *matrixMultipilication_mkl*.

Command: icpc -I/path/to/papi/include -O0 matrixMultiplication.c /path/to/papi/lib/libpapi.a -o mm -fopenmp -mkl

### Usage:
Command: mm matrix_size block_size nthreads

### Expect results:
When set matrix size = 512, block_size = 16, and number of threads = 16,
result should be looked like this: **results/hw22_512_16_16.txt**
The time & floating point operation could be vary from different computers.