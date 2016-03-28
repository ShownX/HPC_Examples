#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "mkl.h"
#include "papi.h"

#define THRESHOLD  32768 /* product size below which matmultleaf is used */  

double matrixMultiplication_noBlock(int matrix_size, int nthreads){
	srand(time(NULL));

	double **A;//[matrix_size][matrix_size];
	double **B;//[matrix_size][matrix_size];
	double **C;//[matrix_size][matrix_size];
	clock_t start_t, end_t;
	double total_t;

	int tid, nthread, i, j, k;
	// allocate the memory
	A = (double **)malloc(sizeof(double *)*matrix_size);
	B = (double **)malloc(sizeof(double *)*matrix_size);
	C = (double **)malloc(sizeof(double *)*matrix_size);
	for(i =0; i < matrix_size; ++i){
		A[i] = (double *)malloc(sizeof(double)*matrix_size);
		B[i] = (double *)malloc(sizeof(double)*matrix_size);
		C[i] = (double *)malloc(sizeof(double)*matrix_size);
	}

	//initialization
	for (i = 0; i < matrix_size; ++i){
		for (j = 0; j < matrix_size; ++j){
			A[i][j] = (rand()%100)/100.0;
			B[i][j] = (rand()%100)/100.0;
			C[i][j] = 0.0;
		}
	}
	
	start_t = clock();
	omp_set_dynamic(0);
	omp_set_num_threads(nthreads);
	#pragma omp parallel shared(A, B, C, nthread) private(tid, i, j, k)
	{
		tid = omp_get_thread_num();
    	if (tid == 0)
     	{
    		nthread = omp_get_num_threads();
    		printf("Starting matrix multiplication with %d threads\n",nthread);
    	}
		/*** Do matrix multiply sharing iterations on outer loop ***/
  		/*** Display who does which iterations for demonstration purposes ***/
  		//printf("Thread %d starting matrix multiply...\n",tid);
  		#pragma omp for schedule(static)
  		//#pragma omp parallel for collapse(2)
		for (i = 0; i < matrix_size; ++i)
		{
			for (j = 0; j < matrix_size; ++j)
			{
				C[i][j] = 0.0;
				for (k = 0; k < matrix_size; ++k)
				{
					C[i][j] = C[i][j] + A[i][k]*B[k][j];
				}
			}
		}
	}
	
	end_t = clock();
	total_t = (double) (end_t - start_t) / CLOCKS_PER_SEC;
	for(i =0; i < matrix_size; ++i){
		free(A[i]);
		free(B[i]);
		free(C[i]);
	}
	free(A);
	free(B);
	free(C);
	return total_t;
}

double matrixMultiplication_Block(int matrix_size, int block_size, int nthreads){
	srand(time(NULL));
	double **A;
	double **B;
	double **C;
	clock_t start_t, end_t;
	double total_t;

	int tid, nthread, i, j, k, l, n;
	
	A = (double **)malloc(sizeof(double *)*matrix_size);
	B = (double **)malloc(sizeof(double *)*matrix_size);
	C = (double **)malloc(sizeof(double *)*matrix_size);
	for(i =0; i < matrix_size; ++i){
		A[i] = (double *)malloc(sizeof(double)*matrix_size);
		B[i] = (double *)malloc(sizeof(double)*matrix_size);
		C[i] = (double *)malloc(sizeof(double)*matrix_size);
	}

	//initialization
	for (i = 0; i < matrix_size; ++i){
		for (j = 0; j < matrix_size; ++j){
			A[i][j] = (rand()%100)/100.0;
			B[i][j] = (rand()%100)/100.0;
			C[i][j] = 0.0;
		}
	}
	
	start_t = clock();
	
	//omp_set_dynamic(0);
	//omp_set_num_threads(nthreads);

	#pragma omp parallel num_threads(nthreads) shared(A, B, C, nthread) private(tid, i, j, k, l, n)
	{
		tid = omp_get_thread_num();
    	if (tid == 0)
     	{
    		nthreads = omp_get_num_threads();
    		printf("Starting matrix multiple example with %d threads\n",nthreads);
    	}
    	//printf("Thread %d starting matrix multiply...\n",tid);
    	for (i = 0; i < matrix_size; i+=block_size)
		{
			for (j = 0; j < matrix_size; j += block_size)
			{
				//#pragma omp parallel for collapse(2)
				#pragma omp for schedule(static)
				for (l = 0; l < block_size; ++l)
				{
					for(n = 0; n < block_size; ++n)
					{
						for (k = 0; k < matrix_size; ++k)
						{
							//#pragma omp critical
							C[i+l][j+n] += A[i+l][k]*B[k][j+k];
						}
					}
				}
			}
		}
	}
	
	end_t = clock();
	total_t = (double) (end_t - start_t) / CLOCKS_PER_SEC;

	for(i =0; i < matrix_size; ++i){
		free(A[i]);
		free(B[i]);
		free(C[i]);
	}
	free(A);
	free(B);
	free(C);

	return total_t;
}

void matmultleaf(double **A, double ** B, double ** C, int mf, int ml, int nf, int nl, int pf, int pl){
/*  
  subroutine that uses the simple triple loop to multiply  
  a submatrix from A with a submatrix from B and store the  
  result in a submatrix of C.   
*/  
// mf, ml; /* first and last+1 i index */  
// nf, nl; /* first and last+1 j index */  
// pf, pl; /* first and last+1 k index */
  	int i, j, k;

  	for (i = 0; i < ml; ++i)
  	{
  		for (j = nf; j < nl; ++j)
  		{
  			C[i][j] = 0.0;
  			for (k = pf; k < pl; ++k)
  			{
  				C[i][j] += A[i][k]*B[k][j];
  			}
  		}
  	}
}

void matmultleaf_block(double **A, double ** B, double ** C, int mf, int ml, int nf, int nl, int pf, int pl, int block_size){
/*  
  subroutine that uses the simple triple loop to multiply  
  a submatrix from A with a submatrix from B and store the  
  result in a submatrix of C.   
*/  
// mf, ml; /* first and last+1 i index */  
// nf, nl; /* first and last+1 j index */  
// pf, pl; /* first and last+1 k index */
  	int i, j, k, l, n;
	if(block_size >= 32){
  		block_size = 8;
	}
  	for (i = 0; i < ml; i += block_size)
  	{
  		for (j = nf; j < nl; j += block_size)
  		{
  			C[i][j] = 0.0;
  			for (l = 0; l < block_size; ++l)
			{
				for(n = 0; n < block_size; ++n)
				{
					for (k = 0; k < pl; ++k)
					{
						C[i+l][j+n] += A[i+l][k]*B[k][j+k];
					}
				}
			}
  		}
  	}
}

void splitMatrix(double **X, int m, double **Y, int mf, int nf){
	int i;
	for (i = 0; i < m; ++i)
	{
		X[i] = &Y[mf+i][nf];
	}
}


void AddMatBlocks(double **T, int m, int n, double **X, double **Y)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			T[i][j] = X[i][j] + Y[i][j];
}

void SubMatBlocks(double **T, int m, int n, double **X, double **Y)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			T[i][j] = X[i][j] - Y[i][j];
}

void StrassenMM(double **A, double **B, double **C, int n, int flag, int block_size){
	if (n*n*n < THRESHOLD)
	{
		if (flag == 1)
		{
			matmultleaf(A, B, C, 0, n, 0, n, 0, n);
		}else{
			matmultleaf_block(A, B, C, 0, n, 0, n, 0, n, block_size);
		}
		
	}
	else
	{
		int m = n/2;
		double ** M1 = (double **)malloc(sizeof(double*)*m);
		double ** M2 = (double **)malloc(sizeof(double*)*m);
		double ** M3 = (double **)malloc(sizeof(double*)*m);
		double ** M4 = (double **)malloc(sizeof(double*)*m);
		double ** M5 = (double **)malloc(sizeof(double*)*m);
		double ** M6 = (double **)malloc(sizeof(double*)*m);
		double ** M7 = (double **)malloc(sizeof(double*)*m);

		double ** wAM1 = (double **)malloc(sizeof(double*)*m);
		double ** wBM1 = (double **)malloc(sizeof(double*)*m);
		double ** wAM2 = (double **)malloc(sizeof(double*)*m);
		double ** wBM3 = (double **)malloc(sizeof(double*)*m);
		double ** wBM4 = (double **)malloc(sizeof(double*)*m);
		double ** wAM5 = (double **)malloc(sizeof(double*)*m);
		double ** wAM6 = (double **)malloc(sizeof(double*)*m);
		double ** wBM6 = (double **)malloc(sizeof(double*)*m);
		double ** wAM7 = (double **)malloc(sizeof(double*)*m);
		double ** wBM7 = (double **)malloc(sizeof(double*)*m);

		double **A11 = new double*[m];
		double **A12 = new double*[m];
		double **A21 = new double*[m];
		double **A22 = new double*[m];

		double **B11 = new double*[m];
		double **B12 = new double*[m];
		double **B21 = new double*[m];
		double **B22 = new double*[m];

		double **C11 = new double*[m];
		double **C12 = new double*[m];
		double **C21 = new double*[m];
		double **C22 = new double*[m];

		int i;
		for(i=0; i<m; i++){
			M1[i] = (double *)malloc(sizeof(double)*m);
			M2[i] = (double *)malloc(sizeof(double)*m);
			M3[i] = (double *)malloc(sizeof(double)*m);
			M4[i] = (double *)malloc(sizeof(double)*m);
			M5[i] = (double *)malloc(sizeof(double)*m);
			M6[i] = (double *)malloc(sizeof(double)*m);
			M7[i] = (double *)malloc(sizeof(double)*m);
			
			wAM1[i] = (double *)malloc(sizeof(double)*m);
			wBM1[i] = (double *)malloc(sizeof(double)*m);
			wAM2[i] = (double *)malloc(sizeof(double)*m);
			wBM3[i] = (double *)malloc(sizeof(double)*m);
			wBM4[i] = (double *)malloc(sizeof(double)*m);
			wAM5[i] = (double *)malloc(sizeof(double)*m);
			wAM6[i] = (double *)malloc(sizeof(double)*m);
			wBM6[i] = (double *)malloc(sizeof(double)*m);
			wAM7[i] = (double *)malloc(sizeof(double)*m);
			wBM7[i] = (double *)malloc(sizeof(double)*m);
		}

		splitMatrix(A11, m, A, 0, 0);
		splitMatrix(A12, m, A, 0, m);
		splitMatrix(A21, m, A, m, 0);
		splitMatrix(A22, m, A, m, m);

		splitMatrix(B11, m, B, 0, 0);
		splitMatrix(B12, m, B, 0, m);
		splitMatrix(B21, m, B, m, 0);
		splitMatrix(B22, m, B, m, m);

		splitMatrix(C11, m, C, 0, 0);
		splitMatrix(C12, m, C, 0, m);
		splitMatrix(C21, m, C, m, 0);
		splitMatrix(C22, m, C, m, m);


		#pragma omp task
		{
		// M1 = (A11 + A22)*(B11 + B22)
		AddMatBlocks(wAM1, m, m, A11, A22);
		AddMatBlocks(wBM1, m, m, B11, B22);
		StrassenMM(wAM1, wBM1, M1, m, flag, block_size);
		}

		#pragma omp task
		{
		//M2 = (A21 + A22)*B11
		AddMatBlocks(wAM2, m, m, A21, A22);
		StrassenMM(wAM2, B11, M2, m, flag, block_size);
		}

		#pragma omp task
		{
		//M3 = A11*(B12 - B22)
		SubMatBlocks(wBM3, m, m, B12, B22);
		StrassenMM(A11, wBM3, M3, m, flag, block_size);
		}

		#pragma omp task
		{
		//M4 = A22*(B21 - B11)
		SubMatBlocks(wBM4, m, m, B21, B11);
		StrassenMM(A22, wBM4, M4, m, flag, block_size);
		}

		#pragma omp task
		{
		//M5 = (A11 + A12)*B22
		AddMatBlocks(wAM5, m, m, A11, A12);
		StrassenMM(wAM5, B22, M5, m, flag, block_size);
		}

		#pragma omp task
		{
		//M6 = (A21 - A11)*(B11 + B12)
		SubMatBlocks(wAM6, m, m, A21, A11);
		AddMatBlocks(wBM6, m, m, B11, B12);
		StrassenMM(wAM6, wBM6, M6, m, flag, block_size);
		}

		#pragma omp task
		{
		//M7 = (A12 - A22)*(B21 + B22)
		SubMatBlocks(wAM7, m, m, A12, A22);
		AddMatBlocks(wBM7, m, m, B21, B22);
		StrassenMM(wAM7, wBM7, M7, m, flag, block_size);
		}
		#pragma omp taskwait
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < m; j++) 
			{
				C11[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
				C12[i][j] = M3[i][j] + M5[i][j];
				C21[i][j] = M2[i][j] + M4[i][j];
				C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
			}
		}


		for(i=0; i<m; i++){
			free(M1[i]);
			free(M2[i]);
			free(M3[i]);
			free(M4[i]);
			free(M5[i]);
			free(M6[i]);
			free(M7[i]);
			
			free(wAM1[i]);
			free(wBM1[i]);
			free(wAM2[i]);
			free(wBM3[i]);
			free(wBM4[i]);
			free(wAM5[i]);
			free(wAM6[i]);
			free(wBM6[i]);
			free(wAM7[i]);
			free(wBM7[i]);
		}
		free(M1);
		free(M2);
		free(M3);
		free(M4);
		free(M5);
		free(M6);
		free(M7);

		free(wAM1);
		free(wBM1);
		free(wAM2);
		free(wBM3);
		free(wBM4);
		free(wAM5);
		free(wAM6);
		free(wBM6);
		free(wAM7);
		free(wBM7);

		delete[] A11; delete[] A12; delete[] A21; delete[] A22;
		delete[] B11; delete[] B12; delete[] B21; delete[] B22;
		delete[] C11; delete[] C12; delete[] C21; delete[] C22;
	}
}

double matrixMultiplication_Recursive(int matrix_size, int block_size, int nthreads, int flag){
	srand(time(NULL));
	double **A;
	double **B;
	double **C;
	clock_t start_t, end_t;
	double total_t;

	int tid, nthread, i, j, k, l, n;
	
	A = (double **)malloc(sizeof(double*)*matrix_size);
	B = (double **)malloc(sizeof(double*)*matrix_size);
	C = (double **)malloc(sizeof(double*)*matrix_size);

	for(i = 0; i < matrix_size; i++){
		A[i] = (double *) malloc(sizeof(double)*matrix_size);
		B[i] = (double *) malloc(sizeof(double)*matrix_size);
		C[i] = (double *) malloc(sizeof(double)*matrix_size);
	}
	//initialization
	for (i = 0; i < matrix_size; ++i){
		for (j = 0; j < matrix_size; ++j){
			A[i][j] = (rand()%100)/100.0;
			B[i][j] = (rand()%100)/100.0;
			C[i][j] = 0.0;
		}
	}
	start_t = clock();
	StrassenMM(A, B, C, matrix_size, flag, block_size);
	end_t = clock();
	total_t = (double) (end_t - start_t) / CLOCKS_PER_SEC;
	return total_t;
}

double matrixMultiplication_mkl(int matrix_size, int nthreads){
	srand(time(NULL));
	double *A, *B, *C;
	double alpha, beta;
	clock_t start_t, end_t;
	double total_t;
	int i;

	A = (double *)mkl_malloc(matrix_size*matrix_size*sizeof(double), 64);
	B = (double *)mkl_malloc(matrix_size*matrix_size*sizeof(double), 64);
	C = (double *)mkl_malloc(matrix_size*matrix_size*sizeof(double), 64);
	
	for(i = 0; i < matrix_size*matrix_size; i++){
		A[i] = (rand()%100)/100.0;
		B[i] = (rand()%100)/100.0;
		C[i] = 0.0;
	}
	start_t = clock();
	omp_set_num_threads(nthreads);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size, matrix_size, alpha, A, matrix_size, B, matrix_size, beta, C, matrix_size);
	end_t = clock();
	total_t = (double) (end_t - start_t) / CLOCKS_PER_SEC;
	return total_t;
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

	int block_size = 10;
	if (argc > 2)
	{
		block_size = atoi(argv[2]);
		if (0 != matrix_size%block_size)
		{
			printf("block_size cannot be divided by the matrix\n");
			return -1;
		}
		printf("Set block size to %d\n", block_size);
	}
	//printf("row is %d\n", matrix_size);
	
	int nthreads = 1;
	if (argc > 3)
	{
		nthreads = atoi(argv[3]);
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
	printf("Start three nested loops - no blocking\n");
	retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
	double total_t = matrixMultiplication_noBlock(matrix_size, nthreads);
	printf("Total time using three nested loops -matrix_size: %d, -threads: %d, -no blocking taken by CPU: %f\n", matrix_size, nthreads, total_t);
	retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
	printf("Real_time:\t%f\nProc_time:\t%f\nTotal flpins:\t%lld\nMFLOPS:\t\t%f\n",
  real_time, proc_time, flpins, mflops);

	printf("\n------------------------------------------------\n");	
	printf("Start three nested loops - blocking\n");
	retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
	total_t = matrixMultiplication_Block(matrix_size, block_size, nthreads);
	printf("Total time using three nested loops -matrix_size: %d, -block size: %d, -thread: %d, taken by CPU: %f\n", matrix_size, block_size, nthreads, total_t);
	retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
	printf("Real_time:\t%f\nProc_time:\t%f\nTotal flpins:\t%lld\nMFLOPS:\t\t%f\n",
  real_time, proc_time, flpins, mflops);

	printf("\n------------------------------------------------\n");
	printf("Start Strassen MM -no blocking\n");
	retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
	total_t = matrixMultiplication_Recursive(matrix_size, block_size, nthreads, 1);
	printf("Total time using Strassen MM -matrix_size: %d, -no blocking, -thread: %d, taken by CPU: %f\n", matrix_size, nthreads, total_t);	
	retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
	printf("Real_time:\t%f\nProc_time:\t%f\nTotal flpins:\t%lld\nMFLOPS:\t\t%f\n",
  real_time, proc_time, flpins, mflops);

	printf("\n------------------------------------------------\n");
	printf("Start Strassen MM - blocking\n");
	retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
	total_t = matrixMultiplication_Recursive(matrix_size, block_size, nthreads, 2);
	printf("Total time using Strassen MM -matrix_size: %d, -block size: 8, -thread: %d, taken by CPU: %f\n", matrix_size, nthreads, total_t);
	retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
	printf("Real_time:\t%f\nProc_time:\t%f\nTotal flpins:\t%lld\nMFLOPS:\t\t%f\n",
  real_time, proc_time, flpins, mflops);

	printf("\n------------------------------------------------\n");
	printf("Start MKL\n");
	retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
	total_t = matrixMultiplication_mkl(matrix_size, nthreads);
	printf("Total time using MKL -matrix_size: %d, -thread: %d, taken by CPU: %f\n", matrix_size, nthreads, total_t);
	retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
	printf("Real_time:\t%f\nProc_time:\t%f\nTotal flpins:\t%lld\nMFLOPS:\t\t%f\n",
  real_time, proc_time, flpins, mflops);
	return 0;
}
