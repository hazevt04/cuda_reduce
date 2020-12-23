// C++ File for utils

#include "my_utils.hpp"

// Just in case there is no intrinsic
// From Hacker's Delight
int my_popcount(unsigned int x) {
   x -= ((x >> 1) & 0x55555555);
   x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
   x = (x + (x >> 4)) & 0x0F0F0F0F;
   x += (x >> 8);
   x += (x >> 16);    
   return x & 0x0000003F;
}

// From Nvidia reduction example
unsigned int next_power_of_two(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

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
