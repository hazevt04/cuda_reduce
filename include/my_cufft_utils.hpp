#pragma once

// My Utility Macros for cuFFT, CUDA's FFT library

#include <cmath>
#include <random>

#include <cufft.h>
#include "my_utils.hpp"

/////////////////////////////
// CUFFT Stuff
/////////////////////////////

// Finally, I can std::cout << cufftComplexValue ?
template<class _CharT, class _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os, const cufftComplex& __c) {
    std::basic_ostringstream<_CharT, _Traits> __s;
    __s.flags(__os.flags());
    __s.imbue(__os.getloc());
    __s.precision(__os.precision());
    __s << "{" << __c.x << ", " << __c.y << "}";
    return __os << __s.str();
}


void gen_cufftComplexes( cufftComplex* complexes, const int num_complexes, const float lower, const float upper ); 

bool cufftComplexes_are_close( const cufftComplex* lvals, const cufftComplex* rvals, 
    const int num_vals, const float max_diff, const std::string& prefix, const bool debug );

void print_cufftComplexes(const cufftComplex* vals,
   const int num_vals,
   const char* prefix,
   const char* delim,
   const char* suffix );

// Why doesnt CUFFT already have something like this in the API?
inline const std::string get_cufft_status_msg(const cufftResult cufft_status) {
  const std::string status_strings[] = {
     "The cuFFT operation was successful\n",
     "cuFFT was passed an invalid plan handle\n",
     "cuFFT failed to allocate GPU or CPU memory\n",
     "No longer used\n",
     "User specified an invalid pointer or parameter\n",
     "Driver or internal cuFFT library error\n",
     "Failed to execute an FFT on the GPU\n",
     "The cuFFT library failed to initialize\n",
     "User specified an invalid transform size\n",
     "No longer used\n",
     "Missing parameters in call\n",
     "Execution of a plan was on different GPU than plan creation\n",
     "Internal plan database error\n",
     "No workspace has been provided prior to plan execution\n",
     "Function does not implement functionality for parameters given.\n",
     "Used in previous versions.\n",
     "Operation is not supported for parameters given.\n" 
   };

   if ( cufft_status < 16 ) {
      return status_strings[cufft_status];
   }
   return "Unknown cufftResult value\n";
}

#define check_cufft_status(cufft_status) { \
  if ( cufft_status != CUFFT_SUCCESS ) { \
    printf( "%s(): ERROR: %s\n", __func__, \
      get_cufft_status_msg( cufft_status ) ); \
    exit(EXIT_FAILURE); \
  } \
}

#define check_cufft_status_error_flag(cufft_status) { \
  if ( cufft_status != CUFFT_SUCCESS ) { \
    printf( "%s(): ERROR: %s\n", __func__, \
      get_cufft_status_msg( cufft_status ) ); \
    error_flag = true; \
    return FAILURE; \
  } \
}

#define check_cufft_status_return(cufft_status) { \
  if ( cufft_status != CUFFT_SUCCESS ) { \
    printf( "%s(): ERROR: %s\n", __func__, \
      get_cufft_status_msg( cufft_status ) ); \
    return FAILURE; \
  } \
}

#define check_cufft_status_throw(cufft_status,loc) { \
  if ( cerror != cudaSuccess ) { \
    throw std::runtime_error{ std::string{ std::string{""#loc ": "} + std::string{get_cufft_status_msg(cufft_status)} + "(" + std::to_string(ccufft_status) + ")" } }; \
  } \
}


#define try_cufft_func(cufft_status, func) { \
  cufft_status = func; \
  check_cufft_status( cufft_status ); \
}

#define try_cufft_func_error_flag(cufft_status, func) { \
  cufft_status = func; \
  check_cufft_status_error_flag( cufft_status ); \
}

#define try_cufft_func_return(cufft_status, func) { \
  cufft_status = func; \
  check_cufft_status_return( cufft_status ); \
}

#define try_cufft_func_throw(cufft_status, func) { \
  cufft_status = func; \
  check_cufft_status_throw( cufft_status, func ); \
}

