#include "my_cufft_utils.hpp"

void gen_cufftComplexes( cufftComplex* complexes, const int num_complexes, const float lower, const float upper ) {
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<float> dist(lower, upper);
   for( int index = 0; index < num_complexes; ++index ) {
      complexes[index].x = dist( mersenne_gen );
      complexes[index].y = dist( mersenne_gen );
   } 
}

bool cufftComplexes_are_close( const cufftComplex* lvals, const cufftComplex* rvals, 
    const int num_vals, const float max_diff, const std::string& prefix, const bool debug ) {

    for( size_t index = 0; index < num_vals; ++index ) {
      float abs_diff_real = abs( lvals[index].x - rvals[index].x );
      float abs_diff_imag = abs( lvals[index].y - rvals[index].y );

      dout << "Index: " << index << ": max_diff = " << max_diff 
         << " actual diffs: { " <<  abs_diff_real << ", " << abs_diff_imag << " }\n";
      if ( ( abs_diff_real > max_diff ) || ( abs_diff_imag > max_diff ) ) {
         dout << "Actual: {" << lvals[index].x << ", " << lvals[index].y << "}\n";
         dout << "Expected: {" << rvals[index].x << ", " << rvals[index].y << "}\n";
         return false;
      }
   }
   return true;  
}

void print_cufftComplexes(const cufftComplex* vals,
   const int num_vals,
   const char* prefix,
   const char* delim,
   const char* suffix ) {

   for (int index = 0; index < num_vals; ++index) {
      std::cout << "\n" << prefix << "Index " << index << ": {" << vals[index].x << ", " << vals[index].y << "}" << ((index == num_vals - 1) ? "\n" : delim);
   }
   std::cout << suffix;
}



// end of C++ file for my_cufft_utils
