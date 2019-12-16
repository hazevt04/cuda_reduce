#include "cuda_utils.h"

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
  if (blockSize >= 64) {
    sdata[tid] += sdata[tid + 32];
  }
  if (blockSize >= 32) { 
    sdata[tid] += sdata[tid + 16];
  }
  if (blockSize >= 16) {
    sdata[tid] += sdata[tid + 8];
  }
  if (blockSize >= 8) {
    sdata[tid] += sdata[tid + 4];
  }
  if (blockSize >= 4) {
    sdata[tid] += sdata[tid + 2];
  }
  if (blockSize >= 2) {
    sdata[tid] += sdata[tid + 1];
  }
}


template <unsigned int blockSize>
__global__ void reduce(int *g_odata, int *g_idata, unsigned int n) {

  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 0;
  while (i < n) { 
    sdata[tid] += g_idata[i] + g_idata[i+blockSize]; 
    i += gridSize; 
  }
  __syncthreads();
  
  if (blockSize >= 512) { 
    if (tid < 256) { 
      sdata[tid] += sdata[tid + 256]; 
    } 
    __syncthreads(); 
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128]; 
    } 
    __syncthreads(); 
  }
  if (blockSize >= 128) { 
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64]; 
    } 
    __syncthreads(); 
  }
  if (tid < 32) {
    warpReduce<blockSize>(sdata, tid);
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}


// From CUDA Handbook, A Comprehensive Guide to GPU Programming by Nicholas Wilt
// (WOW!)
__global__ void reduce_with_atomic( int *out, const int *in, size_t N ) {
  const int tid = threadIdx.x; 
  int partialSum = 0; 
  for ( size_t i = blockIdx.x*blockDim.x + tid; i < N; i += blockDim.x*gridDim.x ) {
    partialSum += in[i]; 
  } 
  atomicAdd( out, partialSum ); 
}


#define BLOCKSIZE 512
#define NUM_STREAMS 1

int main( int argc, char* argv[]) {
  
  int num_vals = 2050;
  int* sums = NULL;
  int* vals = NULL;
  int* d_sums = NULL;
  int* d_vals = NULL;

  size_t num_bytes = num_vals * sizeof( int );

  cudaHostAlloc( (void**)&sums, num_bytes, cudaHostAllocMapped );
  cudaHostAlloc( (void**)&vals, num_bytes, cudaHostAllocMapped );
  
  cudaHostGetDevicePointer( (void**)&d_sums, (void*)sums, 0 );
  cudaHostGetDevicePointer( (void**)&d_vals, (void*)vals, 0 );

  for( int index = 0; index < num_vals; index++ ) {
    sums[index] = 0;
    vals[index] = index+1;
    if ((index < 10) || (index > (num_vals-10))) {
      printf("vals[%d] = %d\n", index, vals[index] );
    }
    if ( index == 11 ) {
      printf( "...\n" ); 
    }
  }
  printf( "\n" ); 
  int exp_sum = (num_vals*(num_vals+1))/2;

  int threads_per_block = BLOCKSIZE;
  int num_blocks = (num_vals + threads_per_block - 1) / threads_per_block;

  printf("num_vals = %d\n", num_vals);
  printf("num_blocks = %d\n\n", num_blocks);

  size_t num_shared_bytes = threads_per_block * sizeof(int);

  reduce<BLOCKSIZE><<<num_blocks, threads_per_block, num_shared_bytes>>>( d_sums, d_vals, num_vals );
  reduce<1><<<1, threads_per_block, num_shared_bytes>>>( d_sums, d_sums, num_vals );

  //cudaStreamSynchronize(streams[0]);
  cudaDeviceSynchronize();

  printf("Sum is %d\n", sums[0] );
  printf("Expected Sum is %d\n", exp_sum );
  if ( sums[0] != exp_sum ) 
    printf( "MISMATCH: expected sum = %d, actual sum = %d\n", exp_sum, sums[0] );
  printf( "\n" ); 

  printf( "Trying reduce with atomicAdd()...\n" ); 
  cudaMemset( sums, 0, sizeof(int) );

  printf("Before reduce_with_atomic, sum is %d\n", sums[0] );

  reduce_with_atomic<<< num_blocks, threads_per_block>>>( sums, vals, num_vals );
  
  cudaDeviceSynchronize();
  printf("Sum from reduce with atomicAdd() is %d\n", sums[0] );
  printf("Expected Sum is %d\n", exp_sum );
  if ( sums[0] != exp_sum ) 
    printf( "MISMATCH: expected sum = %d, reduce with atomicAdd() actual sum = %d\n", exp_sum, sums[0] );
  printf( "\n" ); 
  return SUCCESS;
}
