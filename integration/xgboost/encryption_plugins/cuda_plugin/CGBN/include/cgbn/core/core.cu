/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

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

#include "dispatch_padding.cu"
#include "dispatch_resolver.cu"
#include "dispatch_masking.cu"
#include "dispatch_shift_rotate.cu"
#include "dispatch_dlimbs.cu"

namespace cgbn {

typedef struct {
  // used in gcd
  int32_t alpha_a;
  int32_t alpha_b;
  int32_t beta_a;
  int32_t beta_b;
} signed_coeff_t;

typedef struct {
  // used in modinv
  uint32_t alpha_a;
  uint32_t alpha_b;
  uint32_t beta_a;
  uint32_t beta_b;
} unsigned_coeff_t;

template<class env>
class core_t {
  public:
  static const uint32_t        TPB=env::TPB;
  static const uint32_t        TPI=env::TPI;
  static const uint32_t        BITS=env::BITS;
  static const uint32_t        LIMBS=env::LIMBS;
  static const uint32_t        PADDING=env::PADDING;
  static const uint32_t        MAX_ROTATION=env::MAX_ROTATION;
  static const uint32_t        SHM_LIMIT=env::SHM_LIMIT;
  static const bool            CONSTANT_TIME=env::CONSTANT_TIME;
  static const cgbn_syncable_t SYNCABLE=env::SYNCABLE;  

  static const uint32_t        TPI_ONES=(1ull<<TPI)-1;
  static const uint32_t        PAD_THREAD=(BITS/32)/LIMBS;
  static const uint32_t        PAD_LIMB=(BITS/32)%LIMBS;

  static const uint32_t        DLIMBS=(LIMBS+TPI-1)/TPI;
  static const dlimbs_algs_t   DLIMBS_ALG=(LIMBS<=TPI/2) ? dlimbs_algs_half : (LIMBS<=TPI) ? dlimbs_algs_full : dlimbs_algs_multi;
  
  /* core padding routine */
  
  __device__ __forceinline__ static uint32_t clear_carry(uint32_t &x) {
    return dispatch_padding_t<core_t, PADDING>::clear_carry(x);
  }
  
  __device__ __forceinline__ static uint32_t clear_carry(uint32_t x[LIMBS]) {
    return dispatch_padding_t<core_t, PADDING>::clear_carry(x);
  }
  
  __device__ __forceinline__ static void clear_padding(uint32_t &x) {
    dispatch_padding_t<core_t, PADDING>::clear_padding(x);
  }
  
  __device__ __forceinline__ static void clear_padding(uint32_t x[LIMBS]) {
    dispatch_padding_t<core_t, PADDING>::clear_padding(x);
  }
  
  /* CORE RESOLVER ROUTINES */
  __device__ __forceinline__ static int32_t  fast_negate(uint32_t &x) {
    return dispatch_resolver_t<core_t, TPI, PADDING>::fast_negate(x);
  }
  
  __device__ __forceinline__ static int32_t  fast_negate(uint32_t x[LIMBS]) {
    return dispatch_resolver_t<core_t, TPI, PADDING>::fast_negate(x);
  }

  __device__ __forceinline__ static int32_t  fast_propagate_add(const int32_t carry, uint32_t &x) {
    return dispatch_resolver_t<core_t, TPI, PADDING>::fast_propagate_add(carry, x);
  }
  
  __device__ __forceinline__ static int32_t  fast_propagate_add(const uint32_t carry, uint32_t x[LIMBS]) {
    return dispatch_resolver_t<core_t, TPI, PADDING>::fast_propagate_add(carry, x);
  }

  __device__ __forceinline__ static int32_t  fast_propagate_sub(const uint32_t carry, uint32_t &x) {
    return dispatch_resolver_t<core_t, TPI, PADDING>::fast_propagate_sub(carry, x);
  }
  
  __device__ __forceinline__ static int32_t  fast_propagate_sub(const uint32_t carry, uint32_t x[LIMBS]) {
    return dispatch_resolver_t<core_t, TPI, PADDING>::fast_propagate_sub(carry, x);
  }

  __device__ __forceinline__ static int32_t resolve_add(const uint32_t carry, uint32_t &x) {
    return dispatch_resolver_t<core_t, TPI, PADDING>::resolve_add(carry, x);
  }
  
  __device__ __forceinline__ static int32_t  resolve_add(const uint32_t carry, uint32_t x[LIMBS]) {
    return dispatch_resolver_t<core_t, TPI, PADDING>::resolve_add(carry, x);
  }

  __device__ __forceinline__ static int32_t resolve_sub(const uint32_t carry, uint32_t &x) {
    return dispatch_resolver_t<core_t, TPI, PADDING>::resolve_sub(carry, x);
  }
  
  __device__ __forceinline__ static int32_t  resolve_sub(const uint32_t carry, uint32_t x[LIMBS]) {
    return dispatch_resolver_t<core_t, TPI, PADDING>::resolve_sub(carry, x);
  }


  // HELPER FUNCTIONS FOR GCD AND MOD INV
  __device__ __forceinline__ static void     gcd_product(const uint32_t sync, uint32_t r[LIMBS], const int32_t sa, const uint32_t a[LIMBS], const int32_t sb, const uint32_t b[LIMBS]);
  __device__ __forceinline__ static void     modinv_update_uw_qs(const uint32_t sync, uint32_t r[LIMBS], const uint32_t q, const int32_t s, const uint32_t x[LIMBS]);
  __device__ __forceinline__ static void     modinv_update_ab_sq(const uint32_t sync, uint32_t r[LIMBS], const int32_t shift, const uint32_t q, const uint32_t x[LIMBS]);
  __device__ __forceinline__ static void     modinv_update_uw(const uint32_t sync, uint32_t u[LIMBS], uint32_t w[LIMBS], const unsigned_coeff_t &coeffs);
  __device__ __forceinline__ static bool     modinv_small_delta(const uint32_t sync, uint32_t a[LIMBS], uint32_t b[LIMBS], uint32_t u[LIMBS], uint32_t w[LIMBS], int32_t &delta);

  // DISTRIBUTED LIMB FUNCTIONS FOR DIVISION AND SQUARE ROOT
  __device__ __forceinline__ static void     dlimbs_scatter(uint32_t r[DLIMBS], const uint32_t x[LIMBS], const uint32_t source_thread) {
    dispatch_dlimbs_t<core_t, dlimbs_algs_common>::dlimbs_scatter(r, x, source_thread);
  }
  
  __device__ __forceinline__ static void     dlimbs_gather(uint32_t r[LIMBS], const uint32_t x[DLIMBS], const uint32_t destination_thread) {
    dispatch_dlimbs_t<core_t, dlimbs_algs_common>::dlimbs_gather(r, x, destination_thread);
  }

  __device__ __forceinline__ static void     dlimbs_all_gather(uint32_t r[LIMBS], const uint32_t x[DLIMBS]) {
    dispatch_dlimbs_t<core_t, dlimbs_algs_common>::dlimbs_all_gather(r, x);
  }
  
  __device__ __forceinline__ static uint32_t dlimbs_sqrt_rem_wide(uint32_t s[DLIMBS], uint32_t r[DLIMBS], const uint32_t lo[DLIMBS], const uint32_t hi[DLIMBS]) {
    return dispatch_dlimbs_t<core_t, DLIMBS_ALG>::dlimbs_sqrt_rem_wide(s, r, lo, hi);
  }
  
  __device__ __forceinline__ static void     dlimbs_approximate(uint32_t approx[DLIMBS], const uint32_t denom[DLIMBS]) {
    dispatch_dlimbs_t<core_t, DLIMBS_ALG>::dlimbs_approximate(approx, denom);
  }
  
  __device__ __forceinline__ static void     dlimbs_div_estimate(uint32_t q[DLIMBS], const uint32_t num[DLIMBS], const uint32_t approx[DLIMBS]) {
    dispatch_dlimbs_t<core_t, DLIMBS_ALG>::dlimbs_div_estimate(q, num, approx);
  }
  
  __device__ __forceinline__ static void     dlimbs_sqrt_estimate(uint32_t q[DLIMBS], const uint32_t top, const uint32_t x[DLIMBS], const uint32_t approx[DLIMBS]) {
    dispatch_dlimbs_t<core_t, DLIMBS_ALG>::dlimbs_sqrt_estimate(q, top, x, approx);
  }
  
  
  public:
  __device__ __forceinline__ static uint32_t instance_sync_mask() {
    uint32_t group_thread=threadIdx.x & TPI-1, warp_thread=threadIdx.x & warpSize-1;
    
    return TPI_ONES<<(group_thread ^ warp_thread);
  }
  
  __device__ __forceinline__ static uint32_t sync_mask() {
    // the following is sure to blow up on gcd and modinv and possibly others
    // return (SYNCABLE==cgbn_instance_converged) ? instance_sync_mask() : 0xFFFFFFFF;

    // instead, for now, always use
    return instance_sync_mask();
  }
  
  
  /* BN ROUTINES */
  __device__ __forceinline__ static void     set(uint32_t r[LIMBS], const uint32_t a[LIMBS]) {
    mpset<LIMBS>(r, a);
  }
  __device__ __forceinline__ static void     swap(uint32_t r[LIMBS], uint32_t a[LIMBS]) {
    mpswap<LIMBS>(r, a);
  }
  
  /* BASIC ARITHMETIC */
  __device__ __forceinline__ static int32_t  add(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]);
  __device__ __forceinline__ static int32_t  sub(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]);
  __device__ __forceinline__ static int32_t  negate(uint32_t r[LIMBS], const uint32_t a[LIMBS]);
  __device__ __forceinline__ static void     mul(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);
  __device__ __forceinline__ static void     mul(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t add[LIMBS]);
  __device__ __forceinline__ static void     sqrt_resolve_rem(uint32_t &rem, const uint32_t s, const uint32_t top, const uint32_t r, const uint32_t shift);
  __device__ __forceinline__ static void     sqrt_resolve_rem(uint32_t rem[LIMBS], const uint32_t s[LIMBS], const uint32_t top, const uint32_t r[LIMBS], const uint32_t shift);
  __device__ __forceinline__ static void     sqrt(uint32_t &s, const uint32_t x, const uint32_t numthreads);
  __device__ __forceinline__ static void     sqrt(uint32_t s[LIMBS], const uint32_t x[LIMBS], const uint32_t numthreads);
  __device__ __forceinline__ static void     sqrt_rem(uint32_t &s, uint32_t &r, const uint32_t x, const uint32_t numthreads);
  __device__ __forceinline__ static void     sqrt_rem(uint32_t s[LIMBS], uint32_t r[LIMBS], const uint32_t x[LIMBS], const uint32_t numthreads);
  __device__ __forceinline__ static bool     equals(const uint32_t a[LIMBS], const uint32_t b[LIMBS]);
  __device__ __forceinline__ static int32_t  compare(const uint32_t a[LIMBS], const uint32_t b[LIMBS]);
  
  /* WIDE ARITHMETIC */
  __device__ __forceinline__ static void     mul_wide(uint32_t &lo, uint32_t &hi, const uint32_t a, const uint32_t b, const uint32_t add);
  __device__ __forceinline__ static void     mul_wide(uint32_t lo[LIMBS], uint32_t hi[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t add[LIMBS]);
  __device__ __forceinline__ static void     div_wide(uint32_t &q, const uint32_t lo, const uint32_t hi, const uint32_t denom, const uint32_t numthreads);
  __device__ __forceinline__ static void     div_wide(uint32_t q[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t denom[LIMBS], const uint32_t numthreads);
  __device__ __forceinline__ static void     rem_wide(uint32_t &r, const uint32_t lo, const uint32_t hi, const uint32_t denom, const uint32_t numthreads);
  __device__ __forceinline__ static void     rem_wide(uint32_t r[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t denom[LIMBS], const uint32_t numthreads);
  __device__ __forceinline__ static void     div_rem_wide(uint32_t &q, uint32_t &r, const uint32_t lo, const uint32_t hi, const uint32_t denom, const uint32_t numthreads);
  __device__ __forceinline__ static void     div_rem_wide(uint32_t q[LIMBS], uint32_t r[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t denom[LIMBS], const uint32_t numthreads);
  __device__ __forceinline__ static void     sqrt_wide(uint32_t &s, const uint32_t lo, const uint32_t hi, const uint32_t numthreads);
  __device__ __forceinline__ static void     sqrt_wide(uint32_t s[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t numthreads);
  __device__ __forceinline__ static uint32_t sqrt_rem_wide(uint32_t &s, uint32_t &r, const uint32_t lo, const uint32_t hi, const uint32_t numthreads);
  __device__ __forceinline__ static uint32_t sqrt_rem_wide(uint32_t s[LIMBS], uint32_t r[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t numthreads);

  /* UI32 ROUTINES */
  __device__ __forceinline__ static uint32_t get_ui32(const uint32_t a[LIMBS]);
  __device__ __forceinline__ static void     set_ui32(uint32_t r[LIMBS], const uint32_t value);
  __device__ __forceinline__ static int32_t  add_ui32(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t add);
  __device__ __forceinline__ static int32_t  sub_ui32(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t sub);
  __device__ __forceinline__ static uint32_t mul_ui32(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t mul);
  __device__ __forceinline__ static uint32_t div_ui32(uint32_t &r, const uint32_t a, const uint32_t div);
  __device__ __forceinline__ static uint32_t div_ui32(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t div);
  __device__ __forceinline__ static uint32_t rem_ui32(const uint32_t a, const uint32_t div);
  __device__ __forceinline__ static uint32_t rem_ui32(const uint32_t a[LIMBS], const uint32_t div);
  __device__ __forceinline__ static uint32_t extract_bits_ui32(const uint32_t a[LIMBS], const uint32_t start, const uint32_t len);
  __device__ __forceinline__ static void     insert_bits_ui32(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t start, const uint32_t len, uint32_t value);
  __device__ __forceinline__ static bool     equals_ui32(const uint32_t a[LIMBS], const uint32_t value);
  __device__ __forceinline__ static int32_t  compare_ui32(const uint32_t a[LIMBS], const uint32_t value);
    
  /* LOGICAL */
  __device__ __forceinline__ static void     bitwise_and(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]);
  __device__ __forceinline__ static void     bitwise_ior(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]);
  __device__ __forceinline__ static void     bitwise_xor(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]);
  __device__ __forceinline__ static void     bitwise_complement(uint32_t r[LIMBS], const uint32_t a[LIMBS]);
  __device__ __forceinline__ static void     bitwise_select(uint32_t r[LIMBS], const uint32_t clear[LIMBS], const uint32_t set[LIMBS], const uint32_t select[LIMBS]);

  /* MASKING */
  __device__ __forceinline__ static void     bitwise_mask_copy(uint32_t r[LIMBS], const int32_t numbits) {
    dispatch_masking_t<core_t, PADDING>::bitwise_mask_copy(r, numbits);
  }
  
  __device__ __forceinline__ static void     bitwise_mask_and(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    dispatch_masking_t<core_t, PADDING>::bitwise_mask_and(r, a, numbits);
  }  
  
  __device__ __forceinline__ static void     bitwise_mask_ior(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    dispatch_masking_t<core_t, PADDING>::bitwise_mask_ior(r, a, numbits);
  }
  
  __device__ __forceinline__ static void     bitwise_mask_xor(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    dispatch_masking_t<core_t, PADDING>::bitwise_mask_xor(r, a, numbits);
  }
  
  __device__ __forceinline__ static void     bitwise_mask_select(uint32_t r[LIMBS], const uint32_t clear[LIMBS], const uint32_t set[LIMBS], const int32_t numbits) {
    dispatch_masking_t<core_t, PADDING>::bitwise_mask_select(r, clear, set, numbits);
  }
  
  __device__ __forceinline__ static void     shift_left(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    dispatch_shift_rotate_t<core_t, PADDING>::shift_left(r, a, numbits);
  }
  
  __device__ __forceinline__ static void     shift_right(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    dispatch_shift_rotate_t<core_t, PADDING>::shift_right(r, a, numbits);
  }

  __device__ __forceinline__ static void     rotate_left(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    dispatch_shift_rotate_t<core_t, PADDING>::rotate_left(r, a, numbits);
  }
  
  __device__ __forceinline__ static void     rotate_right(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    dispatch_shift_rotate_t<core_t, PADDING>::rotate_right(r, a, numbits);
  }

  /* BIT COUNTING AND THREAD COUNTING ROUTINES */
  __device__ __forceinline__ static uint32_t pop_count(const uint32_t a[LIMBS]);
  __device__ __forceinline__ static uint32_t clz(const uint32_t a[LIMBS]);
  __device__ __forceinline__ static uint32_t ctz(const uint32_t a[LIMBS]);
  __device__ __forceinline__ static uint32_t clzt(const uint32_t a[LIMBS]);
  __device__ __forceinline__ static uint32_t ctzt(const uint32_t a[LIMBS]);
  
  /* NUMBER THEORETIC ROUTINES */
  __device__ __forceinline__ static void     gcd(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]);
  __device__ __forceinline__ static void     binary_inverse(uint32_t inv[LIMBS], const uint32_t x[LIMBS]);
  __device__ __forceinline__ static bool     modular_inverse(uint32_t inv[LIMBS], const uint32_t x[LIMBS], const uint32_t y[LIMBS]);

  /* MONTGOMERY ROUTINES */
  __device__ __forceinline__ static void     mont_mul(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t n, const uint32_t np0);
  __device__ __forceinline__ static void     mont_mul(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t n[LIMBS], const uint32_t np0);
  __device__ __forceinline__ static void     mont_reduce_wide(uint32_t &r, const uint32_t lo, const uint32_t hi, const uint32_t n, const uint32_t np0, const bool zero);
  __device__ __forceinline__ static void     mont_reduce_wide(uint32_t r[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t n[LIMBS], const uint32_t np0, const bool zero);
};

} /* namespace cgbn */

#include "core_add_sub.cu"
#include "core_short_math.cu"
#include "core_compare.cu"
#include "core_counting.cu"
#include "core_insert_extract.cu"
#include "core_logical.cu"
#include "core_mul.cu"
#include "core_divide_single.cu"
#include "core_divide_multi.cu"
#include "core_sqrt_single.cu"
#include "core_sqrt_multi.cu"
#include "core_gcd.cu"
#include "core_binary_inverse.cu"
#include "core_modular_inverse.cu"
#include "core_mont.cu"

#if defined(XMP_IMAD)
  #include "core_mul_imad.cu"
  #include "core_mont_imad.cu"
#elif defined(XMP_XMAD)
  #include "core_mul_xmad.cu"
  #include "core_mont_xmad.cu"
#elif defined(XMP_WMAD)
  #include "core_mul_wmad.cu"
  #include "core_mont_wmad.cu"
#else
  #warning One of XMP_IMAD, XMP_XMAD, XMP_WMAD must be defined
#endif

