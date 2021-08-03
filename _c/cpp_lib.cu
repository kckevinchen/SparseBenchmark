#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <string>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime_api.h> 
namespace py = pybind11;
#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// #define DEBUG

/**
 * @brief This function time the cuSparse sparse-dense matrix multiplication: A @ B where A is of
 * shape m * k and B is of shape k * n
 * @return float : the time in millisecond
 */

float test_cusparse_gemm(int m, int n, int k, int A_nnz, py::array_t<int> A_csr_offsets, 
	py::array_t<int> A_csr_columns, py::array_t<float>A_csr_values, py::array_t<float> arr_B) {
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
    
	float alpha = 1.0f;
    float beta = 0.0f;
    int ldb = k;
    int ldc = m;		

    //Get the array from the input
    py::buffer_info buf_A_csr_offsets = A_csr_offsets.request();
    py::buffer_info buf_A_csr_columns = A_csr_columns.request();
    py::buffer_info buf_A_csr_values = A_csr_values.request();
    py::buffer_info buf_B_values = arr_B.request();
    int * hA_csr_offsets =(int *) buf_A_csr_offsets.ptr;
    int * hA_csr_columns = (int *) buf_A_csr_columns.ptr;
    float * hA_csr_values = (float *) buf_A_csr_values.ptr;
    float * hB_values = (float *) buf_B_values.ptr;

	//device memory
    int *dA_csr_offsets, *dA_csr_columns;
    float *dA_csr_values, *dB_values, *dC_values;
	int A_num_rows = m;
	int A_num_cols = n;
	int B_num_rows = k;
	int B_num_cols = n;

    //allocate A
    CHECK_CUDA(cudaMalloc((void **)&dA_csr_offsets, (A_num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dA_csr_columns, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dA_csr_values, A_nnz * sizeof(float)));

    //allocate B
    CHECK_CUDA(cudaMalloc((void **)&dB_values, sizeof(float) * B_num_rows * B_num_cols));
    //allocate C
    CHECK_CUDA(cudaMalloc((void **)&dC_values, sizeof(float) * A_num_rows * B_num_cols));

    //to device mtx A
    CHECK_CUDA(cudaMemcpy(dA_csr_offsets, hA_csr_offsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_csr_columns, hA_csr_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_csr_values, hA_csr_values, A_nnz * sizeof(float), cudaMemcpyHostToDevice));
    //to device mtx B
    CHECK_CUDA(cudaMemcpy(dB_values, hB_values, (B_num_rows * B_num_cols) * sizeof(float), cudaMemcpyHostToDevice));
    //create the matrices
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                     dA_csr_offsets, dA_csr_columns, dA_csr_values, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB_values,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL))
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC_values,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL))
    CHECK_CUSPARSE( cusparseCreate(&handle) )		

	//SpGEMM
	Clock::time_point start = Clock::now();
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))

    void *dBuffer = NULL;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    CHECK_CUSPARSE(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))
	cudaDeviceSynchronize();
	Clock::time_point end = Clock::now();
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))

#ifdef DEBUG
    float * hC_values;
    hC_values = (float *) malloc(A_num_rows * B_num_cols * sizeof(float));
    //copy the result to the host
    CHECK_CUDA(cudaMemcpy(hC_values, dC_values, A_num_rows * B_num_cols * sizeof(float), cudaMemcpyDeviceToHost))
    //print out the result
    fprintf(stderr, "printing the multiplication result \n");
    for (int i=0; i<A_num_rows * B_num_cols; i++){
        fprintf(stderr, "%f\n", hC_values[i]);
    }
#endif				
    //device memory free
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA_csr_offsets))
    CHECK_CUDA(cudaFree(dA_csr_columns))
    CHECK_CUDA(cudaFree(dA_csr_values))
    CHECK_CUDA(cudaFree(dB_values))
    CHECK_CUDA(cudaFree(dC_values))
	//get the time
	milliseconds ms = std::chrono::duration_cast<milliseconds> (end-start);
	return ms.count();
}

float test_cublas_sgemm(int m, int n, int k, py::array_t<float> arr_A, py::array_t<float> arr_B) {
// float test_cublas_sgemm(int m, int n, int k, float * arr_A, float * arr_B) {
	//remember the mtx is col based!!!
	//init the variables	
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
	float *A, *B;
	float *d_A, *d_B, *d_C;
#ifdef DEBUG
	//define the output variable C
	float *C;
	C =(float *) malloc(sizeof(float) * m * n);
#endif


 	// get the elements inside the numpy passed in array
	py::buffer_info buf_A = arr_A.request();
	py::buffer_info buf_B = arr_B.request();
	A = (float *) buf_A.ptr;
	B = (float *) buf_B.ptr;
	
	// A = arr_A;
	// B = arr_B;

	//cuda code
	cublasHandle_t handle;
	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "fail handle");
	}

	CHECK_CUDA(cudaMalloc((void **)&d_A, sizeof(float) * m * k))
	CHECK_CUDA(cudaMalloc((void **)&d_B, sizeof(float) * n * k))
	CHECK_CUDA(cudaMalloc((void **)&d_C, sizeof(float) * m * n))

	CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(d_B, B, sizeof(float) * n * k, cudaMemcpyHostToDevice))

	const float a = 1.0,  b = 0.0;
	Clock::time_point start = Clock::now();
	
	cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a, d_A, m, d_B, k, &b, d_C, m);
	cudaDeviceSynchronize();
	Clock::time_point end = Clock::now();

#ifdef DEBUG
	//copy the result back to host memory ofr latter printing
	CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost))
#endif



	CHECK_CUDA(cudaFree(d_A))
	CHECK_CUDA(cudaFree(d_B))
	CHECK_CUDA(cudaFree(d_C))	

	cublasDestroy(handle);

	milliseconds ms = std::chrono::duration_cast<milliseconds> (end-start);

#ifdef DEBUG
	fprintf(stderr, "printing the multiplication result col by col, matrix is %d X %d\n\n", m, n);
	for (int i=0; i<m*n; i++){
		fprintf(stderr, "%f \n", C[i]);
	}
#endif
	return ms.count();
}


//Pybind call
PYBIND11_MODULE(cpp_lib, m){
	m.def("cuBLAS", &test_cublas_sgemm, "the function returning the RT of cuBLAS");
	m.def("cuSPARSE", &test_cusparse_gemm, "the function returning the RT of cuSPARSE");
}

// int main(){
// 	float m = 2.0;
// 	float n = 2.0;
// 	float k = 2.0;
// 	float arr_A[] = {1,2,3,4};
// 	float arr_B[] = {1,0,0,1};
// 	test_cublas_sgemm(m,n,k,arr_A, arr_B);	
// }