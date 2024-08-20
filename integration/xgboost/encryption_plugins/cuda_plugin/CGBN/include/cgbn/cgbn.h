/***

Copyright (c) 2018-2024, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

#ifndef CGBN_H
#define CGBN_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* basic types */
typedef enum {
  cgbn_no_error=0,
  cgbn_unsupported_threads_per_instance=1,
  cgbn_unsupported_size=2,
  cgbn_unsupported_limbs_per_thread=3,
  cgbn_unsupported_operation=4,
  cgbn_threads_per_block_mismatch=5,
  cgbn_threads_per_instance_mismatch=6,
  cgbn_division_by_zero_error=7,
  cgbn_division_overflow_error=8,
  cgbn_invalid_montgomery_modulus_error=9,
  cgbn_modulus_not_odd_error=10,
  cgbn_inverse_does_not_exist_error=11,
} cgbn_error_t;

typedef struct {
  volatile cgbn_error_t _error;
  uint32_t              _instance;
  dim3                  _threadIdx;
  dim3                  _blockIdx;
} cgbn_error_report_t;

typedef enum {
  cgbn_no_checks,       /* disable error checking - improves performance */
  cgbn_report_monitor,  /* writes errors to the reporter object, no other actions */
  cgbn_print_monitor,   /* writes errors to the reporter and prints the error to stdout */
  cgbn_halt_monitor,    /* writes errors to the reporter and halts */
} cgbn_monitor_t;

cudaError_t cgbn_error_report_alloc(cgbn_error_report_t **report);
cudaError_t cgbn_error_report_free(cgbn_error_report_t *report);
bool        cgbn_error_report_check(cgbn_error_report_t *report);
void        cgbn_error_report_reset(cgbn_error_report_t *report);
const char *cgbn_error_string(cgbn_error_report_t *report);

#include "cgbn.cu"

#if defined(__CUDA_ARCH__)
  #if !defined(XMP_IMAD) && !defined(XMP_XMAD) && !defined(XMP_WMAD)
     #if __CUDA_ARCH__<500
       #define XMP_IMAD
     #elif __CUDA_ARCH__<700
       #define XMP_XMAD
     #else
       #define XMP_WMAD
     #endif
  #endif
  #include "cgbn_cuda.h"
#elif defined(__GMP_H__)
  #include "cgbn_mpz.h"
#else
  #include "cgbn_cpu.h"
#endif


template<class env_t, class source_cgbn_t>
__host__ __device__ __forceinline__ void cgbn_set(env_t env, typename env_t::cgbn_t &r, const source_cgbn_t &a) {
  env.set(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_swap(env_t env, typename env_t::cgbn_t &r, typename env_t::cgbn_t &a) {
  env.swap(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_add(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  return env.add(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_sub(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  return env.sub(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_negate(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  return env.negate(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mul(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.mul(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mul_high(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.mul_high(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqr(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  env.sqr(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqr_high(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  env.sqr_high(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_div(env_t env, typename env_t::cgbn_t &q, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom) {
  env.div(q, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_rem(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom) {
  env.rem(r, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_div_rem(env_t env, typename env_t::cgbn_t &q, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom) {
  env.div_rem(q, r, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqrt(env_t env, typename env_t::cgbn_t &s, const typename env_t::cgbn_t &a) {
  env.sqrt(s, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqrt_rem(env_t env, typename env_t::cgbn_t &s, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  env.sqrt_rem(s, r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ bool cgbn_equals(env_t env, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  return env.equals(a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_compare(env_t env, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  return env.compare(a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_extract_bits(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t start, const uint32_t len) {
  env.extract_bits(r, a, start, len);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_insert_bits(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t start, const uint32_t len, const typename env_t::cgbn_t &value) {
  env.insert_bits(r, a, start, len, value);
}


/* ui32 arithmetic routines*/
template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_get_ui32(env_t env, const typename env_t::cgbn_t &a) {
  return env.get_ui32(a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_set_ui32(env_t env, typename env_t::cgbn_t &r, const uint32_t value) {
  env.set_ui32(r, value);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_add_ui32(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t add) {
  return env.add_ui32(r, a, add);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_sub_ui32(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t sub) {
  return env.sub_ui32(r, a, sub);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_mul_ui32(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t mul) {
  return env.mul_ui32(r, a, mul);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_div_ui32(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t div) {
  return env.div_ui32(r, a, div);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_rem_ui32(env_t env, const typename env_t::cgbn_t &a, const uint32_t div) {
  return env.rem_ui32(a, div);
}

template<class env_t>
__host__ __device__ __forceinline__ bool cgbn_equals_ui32(env_t env, const typename env_t::cgbn_t &a, const uint32_t value) {
  return env.equals_ui32(a, value);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_compare_ui32(env_t env, const typename env_t::cgbn_t &a, const uint32_t value) {
  return env.compare_ui32(a, value);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_extract_bits_ui32(env_t env, const typename env_t::cgbn_t &a, const uint32_t start, const uint32_t len) {
  return env.extract_bits_ui32(a, start, len);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_insert_bits_ui32(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t start, const uint32_t len, const uint32_t value) {
  env.insert_bits_ui32(r, a, start, len, value);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_binary_inverse_ui32(env_t env, const uint32_t n0) {
  return env.binary_inverse_ui32(n0);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_gcd_ui32(env_t env, const typename env_t::cgbn_t &a, const uint32_t value) {
  return env.gcd_ui32(a, value);
}


/* wide arithmetic routines */
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mul_wide(env_t env, typename env_t::cgbn_wide_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.mul_wide(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqr_wide(env_t env, typename env_t::cgbn_wide_t &r, const typename env_t::cgbn_t &a) {
  env.sqr_wide(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_div_wide(env_t env, typename env_t::cgbn_t &q, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom) {
  env.div_wide(q, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_rem_wide(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom) {
  env.rem_wide(r, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_div_rem_wide(env_t env, typename env_t::cgbn_t &q, typename env_t::cgbn_t &r, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom) {
  env.div_rem_wide(q, r, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqrt_wide(env_t env, typename env_t::cgbn_t &s, const typename env_t::cgbn_wide_t &a) {
  env.sqrt_wide(s, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqrt_rem_wide(env_t env, typename env_t::cgbn_t &s, typename env_t::cgbn_wide_t &r, const typename env_t::cgbn_wide_t &a) {
  env.sqrt_rem_wide(s, r, a);
}


/* logical, shifting, masking */
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_and(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.bitwise_and(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_ior(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.bitwise_ior(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_xor(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.bitwise_xor(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_complement(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  env.bitwise_complement(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_select(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &clear, const typename env_t::cgbn_t &set, const typename env_t::cgbn_t &select) {
  env.bitwise_select(r, clear, set, select);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_mask_copy(env_t env, typename env_t::cgbn_t &r, const int32_t numbits) {
  env.bitwise_mask_copy(r, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_mask_and(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const int32_t numbits) {
  env.bitwise_mask_and(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_mask_ior(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const int32_t numbits) {
  env.bitwise_mask_ior(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_mask_xor(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const int32_t numbits) {
  env.bitwise_mask_xor(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_mask_select(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &clear, const typename env_t::cgbn_t &set, int32_t numbits) {
  env.bitwise_mask_select(r, clear, set, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_shift_left(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t numbits) {
  env.shift_left(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_shift_right(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t numbits) {
  env.shift_right(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_rotate_left(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t numbits) {
  env.rotate_left(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_rotate_right(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t numbits) {
  env.rotate_right(r, a, numbits);
}


/* bit counting */
template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_pop_count(env_t env, const typename env_t::cgbn_t &a) {
  return env.pop_count(a);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_clz(env_t env, const typename env_t::cgbn_t &a) {
  return env.clz(a);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_ctz(env_t env, const typename env_t::cgbn_t &a) {
  return env.ctz(a);
}


/* accumulator APIs */
template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_resolve(env_t env, typename env_t::cgbn_t &sum, const typename env_t::cgbn_accumulator_t &accumulator) {
  return env.resolve(sum, accumulator);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_set(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const typename env_t::cgbn_t &value) {
  env.set(accumulator, value);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_add(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const typename env_t::cgbn_t &value) {
  env.add(accumulator, value);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sub(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const typename env_t::cgbn_t &value) {
  env.sub(accumulator, value);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_set_ui32(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const uint32_t value) {
  env.set_ui32(accumulator, value);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_add_ui32(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const uint32_t value) {
  env.add_ui32(accumulator, value);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sub_ui32(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const uint32_t value) {
  env.sub_ui32(accumulator, value);
}


/* math */
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_binary_inverse(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &x) {
  env.binary_inverse(r, x);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_gcd(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.gcd(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ bool cgbn_modular_inverse(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &x, const typename env_t::cgbn_t &modulus) {
  return env.modular_inverse(r, x, modulus);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_modular_power(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &x, const typename env_t::cgbn_t &exponent, const typename env_t::cgbn_t &modulus) {
  env.modular_power(r, x, exponent, modulus);
}


/* fast division: common divisor / modulus */
template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_bn2mont(env_t env, typename env_t::cgbn_t &mont, const typename env_t::cgbn_t &bn, const typename env_t::cgbn_t &n) {
  return env.bn2mont(mont, bn, n);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mont2bn(env_t env, typename env_t::cgbn_t &bn, const typename env_t::cgbn_t &mont, const typename env_t::cgbn_t &n, const uint32_t np0) {
  env.mont2bn(bn, mont, n, np0);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mont_mul(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b, const typename env_t::cgbn_t &n, const uint32_t np0) {
  env.mont_mul(r, a, b, n, np0);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mont_sqr(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &n, const uint32_t np0) {
  env.mont_sqr(r, a, n, np0);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mont_reduce_wide(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_wide_t &a, const typename env_t::cgbn_t &n, const uint32_t np0) {
  env.mont_reduce_wide(r, a, n, np0);
}


template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_barrett_approximation(env_t env, typename env_t::cgbn_t &approx, const typename env_t::cgbn_t &denom) {
  return env.barrett_approximation(approx, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_div(env_t env, typename env_t::cgbn_t &q, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_div(q, num, denom, approx, denom_clz);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_rem(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_rem(r, num, denom, approx, denom_clz);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_div_rem(env_t env, typename env_t::cgbn_t &q, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_div_rem(q, r, num, denom, approx, denom_clz);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_div_wide(env_t env, typename env_t::cgbn_t &q, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_div_wide(q, num, denom, approx, denom_clz);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_rem_wide(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_rem_wide(r, num, denom, approx, denom_clz);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_div_rem_wide(env_t env, typename env_t::cgbn_t &q, typename env_t::cgbn_t &r, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_div_rem_wide(q, r, num, denom, approx, denom_clz);
}


/* load/store to global or shared memory */
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_load(env_t env, typename env_t::cgbn_t &r, cgbn_mem_t<env_t::BITS> *const address) {
  env.load(r, address);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_store(env_t env, cgbn_mem_t<env_t::BITS> *address, const typename env_t::cgbn_t &a) {
  env.store(address, a);
}


/* load/store to local memory */
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_load(env_t env, typename env_t::cgbn_t &r, typename env_t::cgbn_local_t *const address) {
  env.load(r, address);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_store(env_t env, typename env_t::cgbn_local_t *address, const typename env_t::cgbn_t &a) {
  env.store(address, a);
}

#endif // CGBN_H
