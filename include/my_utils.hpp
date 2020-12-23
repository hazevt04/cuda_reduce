#pragma once

#define SUCCESS 0
#define FAILURE -2

#include <cstdio>
#include <iostream>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <stdarg.h>
#include <string>
#include <vector>
#include <complex>
#include <algorithm>
#include <random>
#include <exception>
#include <memory>

#include "VariadicToOutputStream.hpp"

#ifndef check_status
#   define check_status(status, msg)                     \
      {                                                  \
         if (status != SUCCESS) {                        \
            printf("%s(): ERROR: " #msg "\n", __func__); \
            exit(EXIT_FAILURE);                          \
         }                                               \
      }
#endif

#ifndef try_new
#   define try_new(type, ptr, num)               \
      {                                          \
         try {                                   \
            ptr = new type[num];                 \
         } catch (const std::bad_alloc& error) { \
            printf(                              \
               "%s(): ERROR: new of %d "         \
               "items for " #ptr " failed\n",    \
               __func__,                         \
               num);                             \
            exit(EXIT_FAILURE);                  \
         }                                       \
      }
#endif


#ifndef try_func
#   define try_func(status, msg, func) \
      {                                \
         status = func;                \
         check_status(status, msg);    \
      }
#endif


#ifndef try_delete
#   define try_delete(ptr) \
      {                    \
         if (ptr)          \
            delete[] ptr;  \
      }
#endif


#ifndef debug_print
#   define debug_printf(debug, fmt, ...) \
      {                                  \
         if (debug) {                    \
            printf(fmt, ##__VA_ARGS__);  \
         }                               \
      }
#endif


#ifndef SWAP
#   define SWAP(a, b) \
      {               \
         (a) ^= (b);  \
         (b) ^= (a);  \
         (a) ^= (b);  \
      }
#endif


#ifndef MAX
#   define MAX(a, b) ((a) > (b)) ? (a) : (b);
#endif


#ifndef CEILING
#   define CEILING(a, b) ((a) + ((b)-1)) / (b);
#endif

// Already included in C++14
template<typename T, typename... Args>
std::unique_ptr<T> my_make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#ifndef dout
#  define dout debug && std::cout
#endif

// Hacker's Delight Second Edition pg 44 ('doz')
// Only valid for signed integers, -2^30 < a,b <=(2^30)-1
// or unsigned integers, 0 < a,b <= (2^31)-1
inline int difference_or_zero(int a, int b) { return ((a - b) & ~((a - b) >> 31)); }


#define MILLISECONDS_PER_SECOND (1000.0f)
typedef std::chrono::steady_clock Steady_Clock;
typedef std::chrono::time_point<std::chrono::steady_clock> Time_Point;
typedef std::chrono::duration<float, std::milli> Duration_ms;
typedef std::chrono::duration<float, std::micro> Duration_us;
typedef std::chrono::duration<float, std::nano> Duration_ns;

// Example usage:
// Time_Point start = Steady_Clock::now();
// Timed code goes here
// Duration_ms duration_ms = Steady_Clock::now() - start;
// printf( "CPU: Func() took %f milliseconds to process %d values\n", duration_ms.count(), num_vals
// );

bool string_is_palindrome(const std::string& s);

template <class T>
void gen_vals(std::vector<T>& vals, const T lower, const T upper) {
   srand(time(NULL));
   T range = upper - lower + (T)1;
   for (auto& val : vals) {
      val = (T)(rand() % (int)range) + lower;
   }
}

template <class T>
void gen_complex_vals(std::complex<T>& vals, const T lower, const T upper, const int num_vals) {
   srand(time(NULL));
   T range = upper - lower + (T)1;
   for (int index = 0; index < num_vals; index++) {
      vals[index].real = (T)(rand() % (int)range) + lower;
      vals[index].imag = (T)(rand() % (int)range) + lower;
   }
}

inline float gen_float( const float lower, const float upper ) {
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<> dist(lower, upper);
   return dist( mersenne_gen );

}

// use for floats or doubles, AKA 'RealType'
template <class RealType>
void gen_reals( RealType* reals, const int num_reals, const float lower, const float upper ) {
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<RealType> dist(lower, upper);
   for( int real_index = 0; real_index < num_reals; ++real_index ) {
      reals[real_index] = dist( mersenne_gen );
   } 
}

// use for floats or doubles, AKA 'RealType'
template <class RealType>
void gen_reals( std::vector<RealType>& reals, const float lower, const float upper ) {
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<RealType> dist(lower, upper);
   for ( auto& real: reals ) {
      real = dist( mersenne_gen );
   } 
}

#include <iomanip>

template <class T>
void print_vals(const std::vector<T>& vals,
   const char* prefix = "",
   const char* delim = " ",
   const char* suffix = "\n") {
   std::cout << prefix;
   std::copy(std::begin(vals), std::end(vals), std::ostream_iterator<T>(std::cout, delim));
   std::cout << suffix;
}

template <class T>
void print_vals(const T* vals,
   const int num_vals,
   const char* prefix = "",
   const char* delim = " ",
   const char* suffix = "\n") {
   for (int index = 0; index < num_vals; ++index) {
      std::cout << "\n" << prefix << "Index " << index << ": " << vals[index] << ((index == num_vals - 1) ? "\n" : delim);
   }
   std::cout << suffix;
}

template <typename T>
void print_vals(const T* vals,
   const int num_vals,
   const int start_index,
   const char* prefix = "",
   const char* delim = " ",
   const char* suffix = "\n") {
   std::cout << prefix;
   for (int index = start_index; index < (start_index + num_vals); ++index) {
      std::cout << prefix << vals[index] << ((index == num_vals - 1) ? "\n" : delim);
   }
   std::cout << suffix;
}


template <typename T>
bool compare_vals(const T* lvals, const T* rvals, int num_vals) {
   for (int index = 0; index < num_vals; ++index) {
      if (lvals[index] != rvals[index]) {
         return false;
      }
   }
   return true;
}


template <typename T>
std::pair<bool, int> mismatch_where(const T* lvals, const T* rvals, int num_vals) {
   for (int index = 0; index < num_vals; ++index) {
      if (lvals[index] != rvals[index]) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         return std::pair<bool, int>{false,index};
      }
   }
   return std::pair<bool, int>{true, -1};
}

// Only here for consistency sake, probably not needed
template <typename T>
bool compare_vals(const std::vector<T>& lvals, const std::vector<T>& rvals) {
   for (int index = 0; index < (int)lvals.size(); ++index) {
      if (lvals[index] != rvals[index]) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         return false;
      }
   }

   return true; 
}

template <typename T>
std::pair<bool, int> mismatch_where(const std::vector<T>& lvals, const std::vector<T>& rvals) {
   for (int index = 0; index < (int)lvals.size(); ++index) {
      if (lvals[index] != rvals[index]) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9)<< lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         return std::pair<bool, int>{false,index};
      }
   }
   return std::pair<bool, int>{true, -1};
}

template<typename T>
using complex_vec = std::vector<std::complex<T>>;


template<typename T>
bool complex_vals_are_close( const complex_vec<T>& lvals, const complex_vec<T>& rvals, const T& max_diff ) {

   for( size_t index = 0; index != lvals.size(); ++index ) {
      T abs_diff_real = abs( lvals[index].real() - rvals[index].real() );
      T abs_diff_imag = abs( lvals[index].imag() - rvals[index].imag() );

      if ( ( abs_diff_real > max_diff ) || ( abs_diff_imag > max_diff ) ) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = {" << lvals[index] << "}\n";
         std::cout << "Rval[" << index << "] = {" << rvals[index] << "}\n";
         std::cout << "Difference = {" << abs_diff_real << ", " << abs_diff_imag << "}\n";
         std::cout << "Max Difference = " << max_diff << "\n";
         return false;
      }
   }
   return true;
}


template<typename T>
std::pair<bool,int> complex_mismatch_where( const complex_vec<T>& lvals, const complex_vec<T>& rvals, const T& max_diff, const std::string prefix="", const bool debug=false ) {

   for( size_t index = 0; index != lvals.size(); ++index ) {
      T abs_diff_real = abs( lvals[index].real() - rvals[index].real() );
      T abs_diff_imag = abs( lvals[index].imag() - rvals[index].imag() );

      if ( ( abs_diff_real > max_diff ) || ( abs_diff_imag > max_diff ) ) {
         std::cout << prefix;
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = {" << lvals[index] << "}\n";
         std::cout << "Rval[" << index << "] = {" << rvals[index] << "}\n";
         std::cout << "Difference = {" << abs_diff_real << ", " << abs_diff_imag << "}\n";
         std::cout << "Max Difference = " << max_diff << "\n";
         return std::pair<bool,int>{false,index};
      }
   }
   dout << __func__ << "(): " << prefix << " All matched.\n";
   return std::pair<bool,int>{true,-1};
}


template<typename T>
bool vals_are_close( const std::vector<T>& lvals, const std::vector<T>& rvals, const T& max_diff, const std::string& prefix="", const bool debug=false ) {

   for( size_t index = 0; index != lvals.size(); ++index ) {
      T abs_diff = abs( lvals[index] - rvals[index] );

      if ( ( abs_diff > max_diff ) ) {
         std::cout << prefix;
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         std::cout << "Difference = " << abs_diff << "\n";
         std::cout << "Max Difference = " << max_diff << "\n";
         return false;
      }
   }
   dout << __func__ << "(): " << prefix << " All matched.\n";
   return true;
}


template<typename T>
bool vals_are_close( const T* lvals, const T* rvals, const int num_vals, const T& max_diff, const std::string& prefix="", const bool debug=false ) {

   dout << __func__ << "(): " << prefix << " num_vals is " << num_vals << "\n";
   if ( lvals == nullptr ) std::cout << __func__ << "(): " << prefix << "lvals is NULL!\n"; 
   if ( rvals == nullptr ) std::cout << __func__ << "(): " << prefix << "rvals is NULL!\n"; 
   for( int index = 0; index < num_vals; ++index ) {
      T abs_diff = abs( lvals[index] - rvals[index] );

      if ( ( abs_diff > max_diff ) ) {
         std::cout << __func__ << "(): " << prefix;
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         std::cout << "Difference = " << abs_diff << "\n";
         std::cout << "Max Difference = " << max_diff << "\n";
         return false;
      }
   }
   dout << __func__ << "(): " << prefix << " All matched.\n";
   return true;
}

template<typename T>
std::pair<bool,int> mismatch_where( const std::vector<T>& lvals, const std::vector<T>& rvals, const T& max_diff, const bool debug ) {

   for( size_t index = 0; index != lvals.size(); ++index ) {
      const T abs_diff = abs( lvals[index] - rvals[index] );

      if ( ( abs_diff > max_diff ) ) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         std::cout << "Difference = " << abs_diff << "\n";
         std::cout << "Max Difference = " << max_diff << "\n";
         return std::pair<bool,int>{false,index};
      }
   }
   return std::pair<bool,int>{true,-1};
}


template<typename T>
std::pair<bool,int> mismatch_where( const T* lvals, const T* rvals, const int num_vals, const T& max_diff, const bool debug ) {

   for( int index = 0; index < num_vals; ++index ) {
      T abs_diff = abs( lvals[index] - rvals[index] );

      if ( ( abs_diff > max_diff ) ) {
         dout << "Mismatch:\n";
         dout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         dout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         dout << "Difference = " << std::setprecision(9) << abs_diff << "\n";
         dout << "Max Difference = " << std::setprecision(9) << max_diff << "\n";
         return std::pair<bool,int>{false,index};
      }
   }
   return std::pair<bool,int>{true,-1};
}


int free_these(void* arg1, ...);

// Example code, not actually meant to be directly used
inline std::string decode_status(int status) {
   char const* status_strings[] = {"This is status 0", "This is status 1"};

   if (status < 2) {
      return std::string(status_strings[status]);
   }
   return std::string("Unknown status value: " + std::to_string(status) + "\n");
}


