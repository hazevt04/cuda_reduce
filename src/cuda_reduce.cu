#include "my_cuda_utils.hpp"

constexpr int BLOCKSIZE = 256;//512;
constexpr int max_threads_per_block = 256;
constexpr int max_num_blocks = 64;

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
__global__ void partial_reduce(int* g_odata, int* g_idata, unsigned int n) {

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
__global__ void single_pass_reduce_with_atomic( int *out, const int *in, size_t N ) {
   const int tid = threadIdx.x; 
   int partialSum = 0; 
   size_t i = blockIdx.x*blockDim.x + tid;
   for ( ; i < N; i += blockDim.x*gridDim.x ) {
      partialSum += in[i]; 
   } 
   atomicAdd( &out[i], partialSum ); 
}


/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};


template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce_nvidia( T *g_odata, T *g_idata, unsigned int n ) {
   // Handle to thread block group
   cg::thread_block cta = cg::this_thread_block();
   T *sdata = SharedMemory<T>();

   // perform first level of reduction,
   // reading from global memory, writing to shared memory
   unsigned int tid = threadIdx.x;
   unsigned int gridSize = blockSize*gridDim.x;

   T mySum = 0;

   // we reduce multiple elements per thread.  The number is determined by the
   // number of active thread blocks (via gridDim).  More blocks will result
   // in a larger gridSize and therefore fewer elements per thread
   if (nIsPow2) {
      unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
      gridSize = gridSize << 1;

      while (i < n) {
         mySum += g_idata[i];
         // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
         if ((i + blockSize) < n) {
            mySum += g_idata[i+blockSize];
         }
         i += gridSize;
      }

   } else {
      unsigned int i = blockIdx.x*blockSize + threadIdx.x;
      while (i < n) {
         mySum += g_idata[i];
         i += gridSize;
      }
   }

   // each thread puts its local sum into shared memory
   sdata[tid] = mySum;
   cg::sync(cta);

   // do reduction in shared mem
   if ((blockSize >= 512) && (tid < 256)) {
      sdata[tid] = mySum = mySum + sdata[tid + 256];
   }

   cg::sync(cta);

   if ((blockSize >= 256) &&(tid < 128)) {
      sdata[tid] = mySum = mySum + sdata[tid + 128];
   }

   cg::sync(cta);

   if ((blockSize >= 128) && (tid <  64)){
      sdata[tid] = mySum = mySum + sdata[tid +  64];
   }

   cg::sync(cta);

   cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

   if (cta.thread_rank() < 32) {
      // Fetch final intermediate sum from 2nd warp
      if (blockSize >=  64) mySum += sdata[tid + 32];
      // Reduce final warp using shuffle
      for (int offset = tile32.size()/2; offset > 0; offset /= 2) {
         mySum += tile32.shfl_down(mySum, offset);
      }
   }

   // write result for this block to global mem
   if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}


int main( int argc, char* argv[] ) {
   try {
      cudaError_t cerror = cudaSuccess;
      bool debug = true;
      int num_vals = 2097152;
      int* sums = nullptr;
      int* vals = nullptr;
      int* d_sums = nullptr;
      int* d_vals = nullptr;

      size_t num_bytes = num_vals * sizeof( int );
      
      int device_id = -1;
      try_cuda_func_throw( cerror, cudaGetDevice( &device_id ) );

      std::unique_ptr<cudaStream_t> stream_ptr = my_make_unique<cudaStream_t>();
      try_cudaStreamCreate( stream_ptr.get() );
      
      try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&sums, num_bytes, cudaHostAllocDefault ) );
      try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&vals, num_bytes, cudaHostAllocDefault ) );

      for( int index = 0; index < num_vals; index++ ) {
         sums[index] = 0;
         vals[index] = 0;
         if (index < 2050) {
            vals[index] = index+1;
         }
         if ( debug ) {
            if ((index < 10) || (index > (num_vals-10))) {
               printf("vals[%d] = %d\n", index, vals[index] );
            }
            if ( index == 11 ) {
               printf( "...\n" ); 
            }
         }
      }
      printf( "\n" ); 

      try_cuda_func_throw( cerror, cudaMalloc( (void**)&d_sums, num_bytes ) );
      try_cuda_func_throw( cerror, cudaMalloc( (void**)&d_vals, num_bytes ) );

      int exp_sum = (num_vals*(num_vals+1))/2;

      int threads_per_block = (num_vals < max_threads_per_block*2) ? next_power_of_two( (num_vals +1)/2 ) : max_threads_per_block; //BLOCKSIZE;
      int num_blocks = (num_vals + ((threads_per_block*2) - 1) ) / (threads_per_block * 2);

      num_blocks = MIN(max_num_blocks, num_blocks);

      if ( debug ) {
         printf( "num_vals = %d\n", num_vals );
         printf( "num_blocks = %d\n", num_blocks );
         printf( "threads_per_block is = %d\n", threads_per_block );
         printf( "BLOCKSIZE is = %d\n", BLOCKSIZE );
         printf( "actual number of threads will be %d\n\n", (num_blocks * threads_per_block) ); 
      }
      size_t num_shared_bytes = threads_per_block * sizeof(int);
      
      //Time_Point start = Steady_Clock::now();

      /////////////////////////////////////////////////
      //// TWO PASS REDUCE
      ////////////////////////////////////////////////////

      //try_cuda_func_throw( cerror, cudaMemcpyAsync( d_vals, vals, num_bytes,
      //         cudaMemcpyHostToDevice, *(stream_ptr.get()) ) );
      
      //partial_reduce<BLOCKSIZE><<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( d_sums, d_vals, num_vals );
      ////try_cuda_func_throw( cerror, cudaPeekAtLastError() );
      //partial_reduce<1><<<1, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( d_sums, d_sums, num_vals );

      //try_cuda_func_throw( cerror, cudaMemcpyAsync( sums, d_sums, num_bytes,
      //         cudaMemcpyDeviceToHost, *(stream_ptr.get()) ) );
      
      //try_cuda_func_throw( cerror, cudaDeviceSynchronize() );

      //Time_Point stop = Steady_Clock::now();
      //Duration_ms duration_ms = stop - start;

      //if ( debug ) {
      //   for( int index = 0; index < num_vals; ++index ) {
      //      printf("After GPU: Sum %d is %d\n", index, sums[index] );
      //   } 
      //   printf("\n\n");
      //   printf("Sum is %d\n", sums[0] );
      //   printf("Expected Sum is %d\n\n", exp_sum );
      //}

      //if ( sums[0] != exp_sum ) {
      //   throw std::runtime_error( std::string{ __func__ } + std::string{"(): MISMATCH: two pass reduce: expected sum = "} +
      //      std::to_string(exp_sum) + std::string{", actual sum = "} + std::to_string( sums[0] )  );
      //}
      //float milliseconds = duration_ms.count();
      //printf( "Two pass reduce with shared memory: All results matched expected. It took %f milliseconds to reduce %d values\n\n", milliseconds, num_vals );


      /////////////////////////////////////////////////
      //// REDUCE WITH ATOMIC ADD
      ////////////////////////////////////////////////////

      //// Clear the sums from the previous run
      //try_cuda_func_throw( cerror, cudaMemset( d_sums, 0, sizeof(int) ) );
      //try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      //for( int index = 0; index < num_vals; ++index ) {
      //   sums[index] = 0;
      //} 
      //num_shared_bytes = threads_per_block * sizeof(int);

      //if ( debug ) printf("Before reduce_with_atomic, sum is %d\n", sums[0] );
   
      //start = Steady_Clock::now();
      
      //try_cuda_func( cerror, cudaMemcpyAsync( d_vals, vals, num_bytes,
      //         cudaMemcpyHostToDevice, *(stream_ptr.get()) ) );
      
      //single_pass_reduce_with_atomic<<< num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( d_sums, d_vals, num_vals );
      //try_cuda_func_throw( cerror, cudaPeekAtLastError() );
      
      //try_cuda_func_throw( cerror, cudaMemcpyAsync( sums, d_sums, num_bytes,
      //         cudaMemcpyDeviceToHost, *(stream_ptr.get()) ) );

      //try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      //stop = Steady_Clock::now();
      //duration_ms = stop - start;

      //if ( debug ) {
      //   printf("Sum from reduce with atomicAdd() is %d\n", sums[0] );
      //   printf("Expected Sum is %d\n\n", exp_sum );
      //}
      //if ( sums[0] != exp_sum ) {
      //   throw std::runtime_error( std::string{ __func__ } + std::string{"(): MISMATCH: reduce with atomicAdd(): expected sum = "} +
      //      std::to_string(exp_sum) + std::string{", actual sum = "} + std::to_string( sums[0] )  );
      //}
      //milliseconds = duration_ms.count();
      //printf( "Single pass reduce with atomicAdd(): All results matched expected. It took %f milliseconds to reduce %d values\n\n", milliseconds, num_vals );

      /////////////////////////////////////////////////////////////////
      // Reduce from NVidia CUDA 11.0 Samples ('reduce6')
      /////////////////////////////////////////////////////////////////
      // Clear the sums from the previous run
      try_cuda_func_throw( cerror, cudaMemset( d_sums, 0, sizeof(int) ) );
      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      for( int index = 0; index < num_vals; ++index ) {
         sums[index] = 0;
      } 
      num_shared_bytes = (threads_per_block <= 32) ? ( 2 * threads_per_block * sizeof(int) ) : ( threads_per_block * sizeof(int) );

      if ( debug ) printf("Before reduce_cg_nvidia, sum is %d\n", sums[0] );

      bool num_vals_is_power_of_two = is_power_of_two( num_vals );

      if ( debug ) {
         printf( "Before reduce_cg_nvidia, num_vals is %s\n", 
            ( num_vals_is_power_of_two ? "a power of two" : "not a power of two") );
      }

      Time_Point start = Steady_Clock::now();
      
      try_cuda_func( cerror, cudaMemcpyAsync( d_vals, vals, num_bytes,
               cudaMemcpyHostToDevice, *(stream_ptr.get()) ) );

      if ( num_vals_is_power_of_two ) {
         reduce_nvidia<int, BLOCKSIZE, true><<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( d_sums, d_vals, num_vals );
      } else {
         reduce_nvidia<int, BLOCKSIZE, false><<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( d_sums, d_vals, num_vals );
      }
      //try_cuda_func_throw( cerror, cudaPeekAtLastError() );
      
      try_cuda_func_throw( cerror, cudaMemcpyAsync( sums, d_sums, num_bytes,
               cudaMemcpyDeviceToHost, *(stream_ptr.get()) ) );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      Time_Point stop = Steady_Clock::now();
      Duration_ms duration_ms = stop - start;

      if ( debug ) {
         printf("Sum from reduce_cg_nvidia is %d\n", sums[0] );
         printf("Expected Sum is %d\n\n", exp_sum );
      }
      if ( sums[0] != exp_sum ) {
         throw std::runtime_error( std::string{ __func__ } + std::string{"(): MISMATCH: reduce_cg_nvidia: expected sum = "} +
            std::to_string(exp_sum) + std::string{", actual sum = "} + std::to_string( sums[0] )  );
      }
      float milliseconds = duration_ms.count();
      printf( "reduce_cg_nvidia: All results matched expected. It took %f milliseconds to reduce %d values\n\n", milliseconds, num_vals );


      if ( sums ) try_cuda_func_throw( cerror, cudaFreeHost( sums ) );
      if ( vals ) try_cuda_func_throw( cerror, cudaFreeHost( vals ) );

      if ( d_sums ) try_cuda_func_throw( cerror, cudaFree( d_sums ) );
      if ( d_vals ) try_cuda_func_throw( cerror, cudaFree( d_vals ) );

      if ( stream_ptr ) try_cuda_func_throw( cerror, cudaStreamDestroy( *(stream_ptr.get()) )  );
      
      return EXIT_SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << "\n"; 
      return EXIT_FAILURE;
   }
}
