# HPC_Examples
High Performance Computing Examples written for the class **UH COSC6365** the Introduction of High Performance Computing.

## mm_acc.c
Comparison of execution time of matrix multiplication using [OpenACC](openacc.org) and [OpenMP](openmp.org)

## How to run code
1. Load mudule PGI using the command `module load pgi`
2. Make
3. Run code

## Test result
When we set the matrix size = 1024 but set the different threads.

| Method | OpenACC | OpenMP 1| OpenMP 2| OpenMP 4| OpenMP 6| OpenMP 8| OpenMP 16|
|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:--------:|
|Time (s)|  0.20   |   2.27  |   1.16  |   0.59  |   0.42  |   0.33  |   0.18   |

