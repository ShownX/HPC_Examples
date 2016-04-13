#include "timer.h"

double matrixMultiplication(int matrix_size, int nthreads){
	srand(time(NULL));

	double *A;//[matrix_size][matrix_size];
	double *B;//[matrix_size][matrix_size];
	double *C;//[matrix_size][matrix_size];
	//clock_t start_t, end_t;
	//double total_t;

	int tid, nthread, i, j, k;
	// allocate the memory
	A = (double **)malloc(sizeof(double *)*matrix_size*matrix_size);
	B = (double **)malloc(sizeof(double *)*matrix_size*matrix_size);
	C = (double **)malloc(sizeof(double *)*matrix_size*matrix_size);

	//initialization
	for (i = 0; i < matrix_size; ++i){
		for (j = 0; j < matrix_size; ++j){
			A[i*matrix_size + j] = (rand()%100)/100.0;
			B[i*matrix_size + j] = (rand()%100)/100.0;
			C[i*matrix_size + j] = 0.0;
		}
	}
	
	StartTimer();
	omp_set_dynamic(0);
	omp_set_num_threads(nthreads);
	#pragma omp parallel shared(A, B, C, nthread) private(tid, i, j, k)
	//#pragma acc kernel compyin(A, B) copy(C)
	#pragma acc parallel loop copyin(A[0:matrix_size*matrix_size], B[0:matrix_size*matrix_size]) copyout(C[0:matrix_size*matrix_size])  
	for (i = 0; i < matrix_size; ++i)
	{
		#pragma omp for schedule(static)
		#pragma acc loop
		for (j = 0; j < matrix_size; ++j)
		{
			double sum = 0;
			for (k = 0; k < matrix_size; ++k)
			{
				sum += A[i*matrix_size + k]*B[k*matrix_size+j];
			}
			C[i*matrix_size + j] = sum;
		}
	}
	
	double runtime = GetTimer();
 
    	printf(" total: %f s\n", runtime / 1000);

	free(A);
	free(B);
	free(C);
	return runtime;
}

int main(int argc, char *argv[]){
	int matrix_size = 100;
	if (argc > 1)
	{
		matrix_size = atoi(argv[1]);
		printf("Set matrix size to %d\n", matrix_size);
	}else{
		printf("Input argument is wrong, pls follow the arguments like: matrix_size block_size, \n");
	}

	int nthreads = 1;
	if (argc > 2)
	{
		nthreads = atoi(argv[2]);
		if (nthreads < 0)
		{
			printf("Set the number of threads is wrong\n");
			return -1;
		}
		printf("Set the number of threads is %d\n", nthreads);
	}

	float real_time, proc_time, mflops;
	long long flpins;
	int retval;
	printf("\n------------------------------------------------\n");
	printf("Start Matrix Multiplication\n");
	double total_t = matrixMultiplication(matrix_size, nthreads);
	printf("Total time for Matrix Mulitiplication with matrix_size: %d, -threads: %d, -no blocking taken by CPU: %f\n", matrix_size, nthreads, total_t);
	
	return 0;
}
