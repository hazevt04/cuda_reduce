
#include <cufft.h>
#include "cuda_utils.h"

char const* get_cufft_status_msg(const cufftResult cufft_status) {
      char const* status_strings[] = {
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
         "Operation is not supported for parameters given.\n" };

   if ( cufft_status < 16 ) {
      return status_strings[cufft_status];
   }
   return "Unknown cufftResult value\n";
}

