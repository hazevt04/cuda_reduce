// C++ File for utils

#include "my_utils.hpp"

// variadic free function!
int free_these(void *arg1, ...) {
    va_list args;
    void *vp;
    if ( arg1 != NULL ) free(arg1);
    va_start(args, arg1);
    while ((vp = va_arg(args, void *)) != 0)
        if ( vp != NULL ) free(vp);
    va_end(args);
    return SUCCESS;
}

void printf_floats( float* const vals, const int num_vals ) {
  for( int index = 0; index < num_vals; index++ ) {
    printf( "%f\n", vals[index] );
  } 
  printf("\n");
}

void printf_ints( int* const vals, const int num_vals ) {
  for( int index = 0; index < num_vals; index++ ) {
    printf( "%d\n", vals[index] );
  } 
  printf("\n");
}

void printf_uints( unsigned int* const vals, const int num_vals ) {
  for( int index = 0; index < num_vals; index++ ) {
    printf( "%u\n", vals[index] );
  } 
  printf("\n");
}

// end of C++ file for utils
