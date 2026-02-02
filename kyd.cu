#include <iostream>
#include <cuda_runtime.h>
#include <string>
inline void CUDA_ERROR_CHECK(cudaError_t err, std::string msg){
  if (err != cudaSuccess) {
    std::cerr << msg << " : " << cudaGetErrorString(err) << std::endl;
  }
}

void printDeviceProperties(cudaDeviceProp& prop, int device) {
  std::cout << "=== CUDA Device:" << device << " ===" << std::endl;
  std::cout << "Name: " << prop.name << std::endl;
  std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "Total Constant Memory: " << prop.totalConstMem /1024.0 << "KB" << std::endl;
  // std::cout << "Clock Rate: " << prop.clockRate / 1000.0 << " MHz" << std::endl;
  //std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
  std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024 * 1024) << " GB" << std::endl;
  std::cout << "Shared Memory per SM: " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
  std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
  std::cout << "Shared Memory per Block for Special Opt-in: " << prop.sharedMemPerBlockOptin / 1024.0 << " KB" << std::endl;
  std::cout << "Registers per SM: " << prop.regsPerMultiprocessor << std::endl;
  std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
  std::cout << "Reserved Shared Memory Per Block: " << prop.reservedSharedMemPerBlock/ 1024.0 << " KB" << std::endl;
  std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Max Blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
  std::cout << "Warp Size: " << prop.warpSize << std::endl;
  std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "Max Block Dimensions: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
  std::cout << "Max Grid Dimensions: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
  std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024.0 << " KB" << std::endl;
  std::cout << "Max Persisting L2 Cache Size: " << prop.persistingL2CacheMaxSize / 1024.0 << " KB" << std::endl;
  std::cout << "ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
  std::cout << "Async Engine Count: " << prop.asyncEngineCount << std::endl;
  std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
  std::cout << "Unified Addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
  std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
  //std::cout << "Peak Memory Bandwidth: " 
    //        << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 
      //      << " GB/s" << std::endl << std::endl;
}

int main() {
    
  int deviceCount;
  cudaDeviceProp deviceProperties;

  CUDA_ERROR_CHECK(cudaGetDeviceCount(&deviceCount), 
      "Failed to get device count");

  std::cout << "Found " << deviceCount << " CUDA capable device(s)" << std::endl << std::endl;
  
  for (int device = 0; device < deviceCount; device++) {
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&deviceProperties, device), 
        "Failed to get device properties");

    printDeviceProperties(deviceProperties, device);
  }
  
  return 0;
}
