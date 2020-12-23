#pragma once

// My Utility Macros for CUDA

#include <cuda_runtime.h>
#include "my_utils.hpp"

// Use for Classes with CUDA (for example)
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

inline void try_cudaStreamCreate( cudaStream_t* pStream ) {
   try {
      cudaError_t cerror = cudaStreamCreate( pStream );
      if ( cerror != cudaSuccess ) {
         std::string err_msg = "cudaStreamCreate() returned error '";
         err_msg += std::string{ cudaGetErrorString( cerror ) } + "'";
         throw std::runtime_error( err_msg );
      }
   } catch( std::exception& ex ) {
      throw;
   }      
}

#define check_cuda_error(cerror,loc) { \
  if ( cerror != cudaSuccess ) { \
    printf( "%s(): "#loc " ERROR: %s (%d)\n", __func__, \
      cudaGetErrorString( cerror ), cerror ); \
    exit(EXIT_FAILURE); \
  } \
}

#define check_cuda_error_flag(cerror,loc) { \
  if ( cerror != cudaSuccess ) { \
    printf( "%s(): "#loc " ERROR: %s (%d)\n", __func__, \
      cudaGetErrorString( cerror ), cerror ); \
    error_flag = true; \
    return FAILURE; \
  } \
}

#define check_cuda_error_return(cerror,loc) { \
  if ( cerror != cudaSuccess ) { \
    printf( "%s(): "#loc " ERROR: %s (%d)\n", __func__, \
      cudaGetErrorString( cerror ), cerror ); \
    return; \
  } \
}

#define check_cuda_error_return_failure(cerror,loc) { \
  if ( cerror != cudaSuccess ) { \
    printf( "%s(): "#loc " ERROR: %s (%d)\n", __func__, \
      cudaGetErrorString( cerror ), cerror ); \
    return FAILURE; \
  } \
}

#define check_cuda_error_throw(cerror,loc) { \
  if ( cerror != cudaSuccess ) { \
    throw std::runtime_error{ std::string{ std::string{""#loc ": "} + std::string{cudaGetErrorString(cerror)} + "(" + std::to_string(cerror) + ")" } }; \
  } \
}

#define try_cuda_func(cerror, func) { \
  cerror = func; \
  check_cuda_error( cerror, func ); \
}

#define try_cuda_func_error_flag(cerror, func) { \
  cerror = func; \
  check_cuda_error_flag( cerror, func ); \
}

#define try_cuda_func_return(cerror, func) { \
  cerror = func; \
  check_cuda_error_return( cerror, func ); \
}

#define try_cuda_func_return_failure(cerror, func) { \
  cerror = func; \
  check_cuda_error_return_failure( cerror, func ); \
}

#define try_cuda_func_throw(cerror, func) { \
  cerror = func; \
  check_cuda_error_throw( cerror, func ); \
}


#define try_cuda_free( cerror, ptr ) { \
  if ((ptr)) { \
    try_cuda_func( (cerror), cudaFree((ptr))); \
    (ptr) = nullptr; \
  } \
}

#define try_cuda_free_host( cerror, ptr ) { \
  if ((ptr)) { \
    try_cuda_func( (cerror), cudaFreeHost((ptr))); \
    (ptr) = nullptr; \
  } \
}

#define try_cuda_free_return( cerror, ptr ) { \
  if (ptr) { \
    try_cuda_func_return( (cerror), cudaFree((ptr)) ); \
  } \
}

#define try_cuda_free_throw( cerror, ptr ) { \
  if (ptr) { \
    try_cuda_func_throw( (cerror), cudaFree((ptr)) ); \
    (ptr) = nullptr; \
  } \
}

#ifdef TRY_FAST_MATH

  #define DIVIDE(quotient, numerator,divisor) { \
    (quotient) = __fdividef((numerator),(divisor)); \
  }
  
  #define DIVIDE_COMPLEX_BY_SCALAR( quotient, numerator, divisor ) { \
     (quotient).x = __fdividef((numerator).x, (divisor)); \
     (quotient).y = __fdividef((numerator).y, (divisor)); \
  }
   
#else

  #define DIVIDE( quotient, numerator, divisor ) { \
    (quotient) = (numerator)/(divisor); \
  }

  #define DIVIDE_COMPLEX_BY_SCALAR( quotient, numerator, divisor ) { \
     (quotient).x = (numerator).x/(divisor); \
     (quotient).y = (numerator).y/(divisor); \
  }

#endif

