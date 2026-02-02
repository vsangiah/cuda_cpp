#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <unistd.h>
#include <cmath>
#include <stdexcept>
#include <string>


// Problem size spec:
const int N = 512;


// Datatype spec:
using real_t = double;

// Tolerance spec:
const real_t REL_TOLERANCE = 1E-16;

// GPU Specific:
const int TILE_SIZE = 16;

/*
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
#define CHECK_CUBLAS_ERROR(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
*/
inline void CHECK_CUDA_ERROR(cudaError_t err) {
  if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
}

inline const char* cublas_get_error_string(cublasStatus_t status) {
    return cublasGetStatusString(status);
}

inline void CHECK_CUBLAS_ERROR(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuBLAS status error: ") + cublas_get_error_string(status)
        );
    }
}

// Kernel 1: Naive matrix multiplication (one thread per output element)
__global__ void naiveMatMul(real_t* A, real_t* B, real_t* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        real_t sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Kernel 2: Tiled matrix multiplication without shared memory
__global__ void tiledMatMul(real_t* A, real_t* B, real_t* C, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    real_t sum = 0.0f;
    if (row < n && col < n) {
        // Process tiles along the shared dimension
        for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
            int tileStart = tile * TILE_SIZE;
            int k_end = min(tileStart + TILE_SIZE, n);
            
            for (int k = tileStart; k < k_end; ++k) {
                sum += A[row * n + k] * B[k * n + col];
            }
        }
        C[row * n + col] = sum;
    }
}

// Kernel 3: Optimized with shared memory tiling
__global__ void sharedMemMatMul(real_t* A, real_t* B, real_t* C, int n) {
    __shared__ real_t sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ real_t sharedB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    real_t sum = 0.0f;
    
    // Process tiles along the shared dimension
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile into shared memory
        if (row < n && tile * TILE_SIZE + tx < n)
            sharedA[ty][tx] = A[row * n + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;
            
        if (tile * TILE_SIZE + ty < n && col < n)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * n + col];
        else
            sharedB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Helper function to initialize matrices with random values
//void initMatrix(real_t* mat, int n, real_t maxVal = 1.0f) {
    //for (int i = 0; i < n * n; ++i) {
        //mat[i] = static_cast<real_t>(rand()) / (static_cast<real_t>(RAND_MAX) / maxVal);
    //}
//}

// Creates random 2D array (M x N matrix)
real_t* createRandomFlatArray(real_t* A, int M, int N, real_t min, real_t max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min, max);
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
          A[i*N+j] = dis(gen);
      }
  }
  return A;
}


// Helper function to verify results (simple checksum)
real_t matrixChecksum(real_t* C, int N) {
  
  real_t sum=0.0;
  for (int i = 0; i <  N*N; ++i) {
        sum += C[i];
    }
    return sum;
}

void matmulCPU(real_t *A, real_t *B, real_t *C, int N){
  
  for(int i=0; i<N; ++i){
    for(int j=0; j<N; ++j){
      real_t sum = 0.0;
      for(int k=0; k<N; ++k){
        sum+=A[i*N+k]*B[k*N+j];
      }
      C[i*N+j] = sum;
    }
  }
}

void printMatrix(real_t *A, int N){
  
  for(int i = 0; i<N; ++i){
      for(int j = 0; j<N; ++j){
        printf("%2.4f    ",A[i*N+j]);
      }
      printf("\n");
    }
}

void compareMatrices(real_t *cpu, real_t *gpu, int N, real_t tol){
  for(int i = 0; i<N; ++i){
    for(int j = 0; j<N; ++j){
      if(std::abs(cpu[i*N+j]-gpu[i*N+j]) > tol){
          printf("Result at C[%d][%d] is not matching within %f \n", i, j, tol);
          }
    }
  }
}

int main() {
    srand(time(NULL));
    
    // Allocate host memory
    int bytes = N*N*sizeof(real_t);
    real_t *h_C = (real_t*)malloc(bytes);
    real_t *C_cpu = (real_t*)malloc(bytes); 
    // Initialize matrices
    real_t minVal=-0.4f, maxVal=0.6f;
    real_t*h_A = (real_t*)malloc(bytes), *h_B = (real_t*)malloc(bytes);
    createRandomFlatArray(h_A, N, N, minVal, maxVal); 
    createRandomFlatArray(h_B, N, N, minVal, maxVal);

    matmulCPU(h_A,h_B, C_cpu, N);
    printf("CPU matmul check sum: %.4f\n", matrixChecksum(C_cpu, N));
    //printf("Sample result of matmulCPU:\n");
    //printMatrix(C_cpu,4);



    // Allocate device memory
    real_t *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, bytes));
    
    
    // Copy matrices to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    // Configure kernel launches
    dim3 blockDimNaive(16, 16);
    dim3 gridDimNaive((N + blockDimNaive.x - 1) / blockDimNaive.x, 
                      (N + blockDimNaive.y - 1) / blockDimNaive.y);
    
    dim3 blockDimTiled(TILE_SIZE, TILE_SIZE);
    dim3 gridDimTiled((N + blockDimTiled.x - 1) / blockDimTiled.x, 
                      (N + blockDimTiled.y - 1) / blockDimTiled.y);
    
    int NUM_TRIALS = 10;

    // Warm-up kernel (for more accurate timing)
    naiveMatMul<<<gridDimNaive, blockDimNaive>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 1. Naive kernel
    cudaEventRecord(start);
    for(int i = 0; i<NUM_TRIALS; ++i){
      naiveMatMul<<<gridDimNaive, blockDimNaive>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Naive kernel const time: %.4f ms\n", milliseconds/NUM_TRIALS);
    
    // Copy result back for verification
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    
    printf("Naive result checksum: %.4f\n", matrixChecksum(h_C, N));
    //compareMatrices(C_cpu, h_C, N, REL_TOLERANCE);
    // 2. Tiled kernel (without shared memory)
    cudaEventRecord(start);
    for(int i = 0; i<NUM_TRIALS; ++i){
      tiledMatMul<<<gridDimTiled, blockDimTiled>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiled kernel time: %.4f ms\n", milliseconds/NUM_TRIALS);
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    
    printf("Tiled result checksum: %.4f\n", matrixChecksum(h_C, N));  
    //compareMatrices(C_cpu, h_C, N, REL_TOLERANCE);
    
    // 3. Shared memory kernel
    cudaEventRecord(start);
    for(int i = 0; i<NUM_TRIALS; ++i){
    sharedMemMatMul<<<gridDimTiled, blockDimTiled>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Shared memory kernel time: %.4f ms\n", milliseconds/NUM_TRIALS);
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    
    printf("Shared memory result checksum: %.4f\n", matrixChecksum(h_C, N));
    //compareMatrices(C_cpu, h_C, N, REL_TOLERANCE);
    
    // 4. cuBLAS implementation
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    
    const real_t alpha = 1.0;
    const real_t beta = 0.0;
    
    cudaEventRecord(start);
    for(int i = 0; i<NUM_TRIALS; ++i){  
    CHECK_CUBLAS_ERROR(cublasDgemm(handle, 
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_B, N,  // Note: cuBLAS uses column-major, so we swap A and B
                             d_A, N,
                             &beta,
                             d_C, N));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuBLAS time: %.4f ms\n", milliseconds/NUM_TRIALS);
    
    // Copy cuBLAS result for verification
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    printf("cuBLAS result checksum: %.4f\n", matrixChecksum(h_C, N));
    //compareMatrices(C_cpu, h_C, N, REL_TOLERANCE);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(C_cpu);
    free(h_A);
    free(h_B);
    free(h_C);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
