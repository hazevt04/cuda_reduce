#include "my_cuda_utils.hpp"

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

// Must be called twice. The second time with 1 block and 'num_blocks' threads
template <unsigned int blockSize>
__global__ void reduce(int *g_odata, int *g_idata, unsigned int n) {

   extern __shared__ int sdata[];
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*(blockSize*2) + tid;
   unsigned int gridSize = blockSize*2*gridDim.x;

   sdata[tid] = 0;
   __syncthreads();

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
   __syncthreads(); 

   if (tid == 0) {
      g_odata[blockIdx.x] = sdata[0];
   }
}

// SINGLE PASS REDUCTION (Only if the XOR operator is supported natively by 
// an atomic operator in HW). The output must be initialized to zero before calling
// this kernel! The kernel cannot do this because 'CUDA's execution model does 
// not enable the race condition to be resolved between thread blocks'
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

int main( int argc, char* argv[] ) {
   try {
      cudaError_t cerror = cudaSuccess;

      int num_vals = 2050;
      int* sums = nullptr;
      int* vals = nullptr;
      int* d_sums = nullptr;
      int* d_vals = nullptr;

      size_t num_bytes = num_vals * sizeof( int );

      try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&sums, num_bytes, cudaHostAllocMapped ) );
      try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&vals, num_bytes, cudaHostAllocMapped ) );

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

      try_cuda_func_throw( cerror, cudaHostGetDevicePointer( (void**)&d_sums, (void*)sums, 0 ) );
      try_cuda_func_throw( cerror, cudaHostGetDevicePointer( (void**)&d_vals, (void*)vals, 0 ) );

      printf( "\n" ); 
      int exp_sum = (num_vals*(num_vals+1))/2;

      int threads_per_block = BLOCKSIZE;
      int num_blocks = (num_vals + threads_per_block - 1) / threads_per_block;

      printf("num_vals = %d\n", num_vals);
      printf("num_blocks = %d\n\n", num_blocks);

      size_t num_shared_bytes = threads_per_block * sizeof(int);
      Time_Point start = Steady_Clock::now();

      reduce<BLOCKSIZE><<<num_blocks, threads_per_block, num_shared_bytes>>>( d_sums, d_vals, num_vals );
      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );

      printf("Before reduce<1>(), Sum is %d\n", sums[0] );

      reduce<1><<<1, threads_per_block, num_shared_bytes>>>( d_sums, d_vals, num_vals );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      Time_Point stop = Steady_Clock::now();
      Duration_ms duration_ms = stop - start;

      printf("Sum is %d\n", sums[0] );
      printf("Expected Sum is %d\n", exp_sum );
      if ( sums[0] != exp_sum ) {
         throw std::runtime_error( std::string{ __func__ } + std::string{"(): MISMATCH: two call reduce: expected sum = "} +
            std::to_string(exp_sum) + std::string{", actual sum = "} + std::to_string( sums[0] )  );
      }
      printf( "\n" ); 
      float milliseconds = duration_ms.count();
      printf( "Two pass reduce with shared memory took %f milliseconds to reduce %d values\n\n", milliseconds, num_vals );


      printf( "Trying reduce with atomicAdd()...\n" ); 
      try_cuda_func_throw( cerror, cudaMemset( sums, 0, sizeof(int) ) );
      printf("Before reduce_with_atomic, sum is %d\n", sums[0] );

      start = Steady_Clock::now();
      reduce_with_atomic<<< num_blocks, threads_per_block>>>( sums, vals, num_vals );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      stop = Steady_Clock::now();
      duration_ms = stop - start;

      printf("Sum from reduce with atomicAdd() is %d\n", sums[0] );
      printf("Expected Sum is %d\n", exp_sum );
      if ( sums[0] != exp_sum ) {
         throw std::runtime_error( std::string{ __func__ } + std::string{"(): MISMATCH: reduce with atomicAdd(): expected sum = "} +
            std::to_string(exp_sum) + std::string{", actual sum = "} + std::to_string( sums[0] )  );
      }
      printf( "\n" ); 
      milliseconds = duration_ms.count();
      printf( "Single pass reduce with atomicAdd() took %f milliseconds to reduce %d values\n\n", milliseconds, num_vals );

      return EXIT_SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << "\n"; 
      return EXIT_FAILURE;
   }
}
