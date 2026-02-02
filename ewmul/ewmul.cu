#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <unistd.h>
// Creates random vector (1D array)
float* createRandomVector(int M, float min, float max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  
  float* v = new float[M];
  for (int i = 0; i < M; i++) {
      v[i] = dis(gen);
  }
  return v;
}

// Creates random 2D array (M x N matrix)
float** createRandomArray(int M, int N, float min, float max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  
  float** A = new float*[M];
  for (int i = 0; i < M; i++) {
      A[i] = new float[N];
      for (int j = 0; j < N; j++) {
          A[i][j] = dis(gen);
      }
  }
  return A;
}


void vecElemWiseMultiplyHost(float* h_A, float* h_B, float* h_C, int N)
{
	for(int i=0; i<N; ++i)
	{
		h_C[i]=h_A[i]*h_B[i];
	}

}

__global__ void vecElemwiseMultiplyKernel(float* d_A, float* d_B, float* d_C, int N)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		d_C[idx] = d_A[idx] * d_B[idx];
	}	
}

void vecElemwiseMulitplyDeviceSingleStream(float* h_A, float* h_B, float* h_C, int N)
{
	int size = N*sizeof(float);
	float *d_A, *d_B, *d_C;
  
  // Experiment with block sizes
  int gridSize, blockSize=256;
  //int nBlockSizes = sizeof(blockSizes)/sizeof(blockSizes[0]);
	
  // Allocate device memory d_A (+2)
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	// Copy h_A to d_A (+2)
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	
	// Launch addition kernel to evaluate d_C
	// <<<numOfBlocks, numOfThreadsPerBlock>>>

  //for(int i=0; i<nBlockSizes; ++i)
  //{   
    //blockSize = blockSizes[i];
    gridSize = ceil(N/blockSize);
	 
    for(int j=0; j<20;++j) 
    {    
      vecElemwiseMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    } 
    // Synchronize the host with the device to ensure completion
    // cudaError_t err = cudaDeviceSynchronize();
    // cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //   std::cout<<"There is some error";
    // }
  //}

  cudaDeviceSynchronize();

	// Copy device d_C to host h_C
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	// Free up d_A (+2)
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}

void vecElemwiseMulitplyDeviceMulitpleStreams(float* h_A, float* h_B, float* h_C, int N, int numStreams)
{
  int size = N*sizeof(float);
	float *d_A, *d_B, *d_C;

  cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

  cudaStream_t *cuStreams = new cudaStream_t[numStreams];
  for(int j=0; j<numStreams; ++j)
  {
    cudaStreamCreate(&cuStreams[j]);
  }
  
  int numElementsPerStream = N/numStreams;
  int bytesPerStream = sizeof(float)*numElementsPerStream;
  

  int blockSize = 256;
  int numGrids = ceil(numElementsPerStream/blockSize);

  for(int i=0; i<numStreams; ++i)
  {
    int indexOffset = i*numElementsPerStream;
    cudaMemcpyAsync(&d_A[indexOffset], &h_A[indexOffset], bytesPerStream, cudaMemcpyDeviceToHost, cuStreams[i]);
    cudaMemcpyAsync(&d_B[indexOffset], &h_B[indexOffset], bytesPerStream, cudaMemcpyDeviceToHost, cuStreams[i]);

    vecElemwiseMultiplyKernel<<<numGrids, blockSize, 0, cuStreams[i]>>>(&d_A[indexOffset],      
      &d_B[indexOffset], 
      &d_C[indexOffset], 
      numElementsPerStream);  
  }

  for(int i=0; i<numStreams; ++i)
  {
    int indexOffset = i*numElementsPerStream;
    cudaStreamSynchronize(cuStreams[i]);    
    cudaMemcpyAsync(&h_C[indexOffset], &d_C[indexOffset], size, cudaMemcpyDeviceToHost, cuStreams[i]);
  }
  cudaDeviceSynchronize();
  for(int i=0; i<numStreams; ++i)
  {
    cudaStreamDestroy(cuStreams[i]);
  }
  // Try: cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  delete[] cuStreams; 
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
int main()
{
	
	const int N = 1024*1024;
	const float minVal = -452.53461;
	const float maxVal = 546.08427;
	float *A = createRandomVector(N,minVal,maxVal);
	float *B = createRandomVector(N,minVal,maxVal);
	float C[N] = {};
	
	//vecAddHost(A,B,C,N);
	
	vecElemwiseMulitplyDeviceSingleStream(A,B,C,N);

  //sleep(1);

  //int streamSizes[] = {1}; // number of elements per stream
  //int numStreamExperiments = sizeof(streamSizes) / sizeof(streamSizes[0]);
  
  //for(int i=0; i<numStreamExperiments; ++i)
  //{
    //int numStreams = streamSizes[i];
    //vecElemwiseMulitplyDeviceMulitpleStreams(A,B,C,N,numStreams);
   // sleep(1);
  //}

	// for(int i=0; i<N ; ++i)
	// {
	// 	std::cout<<A[i]<<" + "<<B[i]<<" = "<<C[i]<<"\n";
	// }

	delete A, B, C;

	return 0;
}
