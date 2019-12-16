#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

// Utility Macros for CUDA

#include <cufft.h>

#include "utils.h"

#define check_cuda_error(cerror,loc) { \
  if ( cerror != cudaSuccess ) { \
    printf( "%s(): "#loc " ERROR: %s (%d)\n", __func__, \
      cudaGetErrorString( cerror ), cerror ); \
    exit(EXIT_FAILURE); \
  } \
}

#define try_cuda_func(cerror, func) { \
  cerror = func; \
  check_cuda_error( cerror, func ); \
}

#define try_cuda_free( cerror, ptr, free_func ) { \
  if (ptr) { \
    try_cuda_func( (cerror), (free_func) ); \
  } \
}

#ifdef TRY_FAST_MATH

  #define DIVIDE(q, n,d) { \
    (q) = __fdividef((n),(d)); \
  }

#else

  #define DIVIDE(q, n,d) { \
    (q) = (n)/(d); \
  }

#endif

/////////////////////////////
// CUFFT Stuff
/////////////////////////////

// Returns string based on the cuffResult value returned by a CUFFT call
// Why doesnt CUFFT already have something like this in the API?
char const* get_cufft_status_msg(const cufftResult cufft_status);

#define check_cufft_status(cufft_status) { \
  if ( cufft_status != CUFFT_SUCCESS ) { \
    printf( "%s(): ERROR: %s\n", __func__, \
      get_cufft_status_msg( cufft_status ) ); \
    exit(EXIT_FAILURE); \
  } \
}



#define try_cufft_func(cufft_status, func) { \
  cufft_status = func; \
  check_cufft_status( cufft_status ); \
}
#endif // ifndef _CUDA_UTILS_H_
