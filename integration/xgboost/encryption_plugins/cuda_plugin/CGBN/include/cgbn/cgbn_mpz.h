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

#include <stdio.h>
#include <stdlib.h>

#if !defined(__CUDACC__)
  typedef struct {uint32_t x; uint32_t y; uint32_t z;} dim3;
  #define __host__
#endif

typedef enum {
  cgbn_instance_converged,
  cgbn_warp_converged,
  cgbn_block_converged,
  cgbn_grid_converged,
} cgbn_convergence_t;

class cgbn_default_parameters_t {
  public:
  static const uint32_t TPB=0;
};

/* forward declarations */
template<uint32_t tpi, class params>
class cgbn_context_t;

template<class context_t, uint32_t bits, cgbn_convergence_t convergence>
class cgbn_env_t;

template<uint32_t bits>
struct cgbn_mem_t {
  public:
  uint32_t _limbs[(bits+31)/32];
};

template<uint32_t tpi, class params=cgbn_default_parameters_t>
class cgbn_context_t {
  public:
  static const uint32_t  TPI=tpi;

  const cgbn_monitor_t   _monitor;
  cgbn_error_report_t   *const _report;
  int32_t                _instance;

  public:
  __host__ cgbn_context_t();
  __host__ cgbn_context_t(cgbn_monitor_t monitor);
  __host__ cgbn_context_t(cgbn_monitor_t monitor, cgbn_error_report_t *report);
  __host__ cgbn_context_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, uint32_t instance);
  __host__ bool check_errors() const;
  __host__ void report_error(cgbn_error_t error) const;

  template<class env_t>
  env_t env() {
    env_t env(*this);

    return env;
  }

  template<uint32_t bits, cgbn_convergence_t convergence>
  cgbn_env_t<cgbn_context_t, bits, convergence> env() {
    cgbn_env_t<cgbn_context_t, bits, convergence> env(*this);

    return env;
  }
};

template<class context_t, uint32_t bits, cgbn_convergence_t convergence=cgbn_instance_converged>
class cgbn_env_t {
  public:
  static const uint32_t TPI=context_t::TPI;
  static const uint32_t BITS=bits;
  static const uint32_t LIMBS=(bits/32+TPI-1)/TPI;
  static const uint32_t LOCAL_LIMBS=((bits+32)/64+TPI-1)/TPI*TPI;
  static const uint32_t UNPADDED_BITS=TPI*LIMBS*32;

  struct cgbn_t {
    public:
    typedef cgbn_env_t parent_env_t;
    mpz_t _z;

    __host__ cgbn_t() {
      mpz_init(_z);
    }

    __host__ ~cgbn_t() {
      mpz_clear(_z);
    }
  };
  struct cgbn_wide_t {
    public:
    cgbn_t _low, _high;
  };
  struct cgbn_local_t {
    public:
    mpz_t _z;

    __host__ cgbn_local_t() {
      mpz_init(_z);
    }

    __host__ ~cgbn_local_t() {
      mpz_clear(_z);
    }
  };
  struct cgbn_accumulator_t {
    public:
    mpz_t _z;

    __host__ cgbn_accumulator_t() {
      mpz_init(_z);
    }

    __host__ ~cgbn_accumulator_t() {
      mpz_clear(_z);
    }
  };

  const context_t &_context;

  __host__ cgbn_env_t(const context_t &context);

  /* size conversion */
  template<typename source_cgbn_t>
  __host__ void       set(cgbn_t &r, const source_cgbn_t &source) const;

  /* set/get routines */
  __host__ void       set(cgbn_t &r, const cgbn_t &a) const;
  __host__ void       swap(cgbn_t &r, cgbn_t &a) const;
  __host__ void       extract_bits(cgbn_t &r, const cgbn_t &a, uint32_t start, uint32_t len) const;
  __host__ void       insert_bits(cgbn_t &r, const cgbn_t &a, uint32_t start, uint32_t len, const cgbn_t &value) const;

  /* ui32 arithmetic routines*/
  __host__ uint32_t   get_ui32(const cgbn_t &a) const;
  __host__ void       set_ui32(cgbn_t &r, const uint32_t value) const;
  __host__ int32_t    add_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t add) const;
  __host__ int32_t    sub_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t sub) const;
  __host__ uint32_t   mul_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t mul) const;
  __host__ uint32_t   div_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t div) const;
  __host__ uint32_t   rem_ui32(const cgbn_t &a, const uint32_t div) const;
  __host__ bool       equals_ui32(const cgbn_t &a, const uint32_t value) const;
  __host__ int32_t    compare_ui32(const cgbn_t &a, const uint32_t value) const;
  __host__ uint32_t   extract_bits_ui32(const cgbn_t &a, const uint32_t start, const uint32_t len) const;
  __host__ void       insert_bits_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t start, const uint32_t len, const uint32_t value) const;
  __host__ uint32_t   binary_inverse_ui32(const uint32_t n0) const;
  __host__ uint32_t   gcd_ui32(const cgbn_t &a, const uint32_t value) const;

  /* bn arithmetic routines */
  __host__ int32_t    add(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __host__ int32_t    sub(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __host__ int32_t    negate(cgbn_t &r, const cgbn_t &a) const;
  __host__ void       mul(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __host__ void       mul_high(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __host__ void       sqr(cgbn_t &r, const cgbn_t &a) const;
  __host__ void       sqr_high(cgbn_t &r, const cgbn_t &a) const;
  __host__ void       div(cgbn_t &q, const cgbn_t &num, const cgbn_t &denom) const;
  __host__ void       rem(cgbn_t &r, const cgbn_t &num, const cgbn_t &denom) const;
  __host__ void       div_rem(cgbn_t &q, cgbn_t &r, const cgbn_t &num, const cgbn_t &denom) const;
  __host__ void       sqrt(cgbn_t &s, const cgbn_t &a) const;
  __host__ void       sqrt_rem(cgbn_t &s, cgbn_t &r, const cgbn_t &a) const;
  __host__ bool       equals(const cgbn_t &a, const cgbn_t &b) const;
  __host__ int32_t    compare(const cgbn_t &a, const cgbn_t &b) const;

  /* wide math routines */
  __host__ void       mul_wide(cgbn_wide_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __host__ void       sqr_wide(cgbn_wide_t &r, const cgbn_t &a) const;
  __host__ void       div_wide(cgbn_t &q, const cgbn_wide_t &num, const cgbn_t &denom) const;
  __host__ void       rem_wide(cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom) const;
  __host__ void       div_rem_wide(cgbn_t &q, cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom) const;
  __host__ void       sqrt_wide(cgbn_t &s, const cgbn_wide_t &a) const;
  __host__ void       sqrt_rem_wide(cgbn_t &s, cgbn_wide_t &r, const cgbn_wide_t &a) const;

  /* logical, shifting, masking */
  __host__ void       bitwise_complement(cgbn_t &r, const cgbn_t &a) const;
  __host__ void       bitwise_and(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __host__ void       bitwise_ior(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __host__ void       bitwise_xor(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __host__ void       bitwise_select(cgbn_t &r, const cgbn_t &clear, const cgbn_t &set, const cgbn_t &select) const;
  __host__ void       bitwise_mask_copy(cgbn_t &r, const int32_t numbits) const;
  __host__ void       bitwise_mask_and(cgbn_t &r, const cgbn_t &a, const int32_t numbits) const;
  __host__ void       bitwise_mask_ior(cgbn_t &r, const cgbn_t &a, const int32_t numbits) const;
  __host__ void       bitwise_mask_xor(cgbn_t &r, const cgbn_t &a, const int32_t numbits) const;
  __host__ void       bitwise_mask_select(cgbn_t &r, const cgbn_t &clear, const cgbn_t &set, const int32_t numbits) const;
  __host__ void       shift_left(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const;
  __host__ void       shift_right(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const;
  __host__ void       rotate_left(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const;
  __host__ void       rotate_right(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const;

  /* faster shift and rotate by a constant number of bits */
  template<uint32_t numbits> __host__ void shift_left(cgbn_t &r, const cgbn_t &a) const;
  template<uint32_t numbits> __host__ void shift_right(cgbn_t &r, const cgbn_t &a) const;
  template<uint32_t numbits> __host__ void rotate_left(cgbn_t &r, const cgbn_t &a) const;
  template<uint32_t numbits> __host__ void rotate_right(cgbn_t &r, const cgbn_t &a) const;

  /* bit counting */
  __host__ uint32_t   pop_count(const cgbn_t &a) const;
  __host__ uint32_t   clz(const cgbn_t &a) const;
  __host__ uint32_t   ctz(const cgbn_t &a) const;

  /* accumulator APIs */
  __host__ int32_t    resolve(cgbn_t &sum, const cgbn_accumulator_t &accumulator) const;
  __host__ void       set_ui32(cgbn_accumulator_t &accumulator, const uint32_t value) const;
  __host__ void       add_ui32(cgbn_accumulator_t &accumulator, const uint32_t value) const;
  __host__ void       sub_ui32(cgbn_accumulator_t &accumulator, const uint32_t value) const;
  __host__ void       set(cgbn_accumulator_t &accumulator, const cgbn_t &value) const;
  __host__ void       add(cgbn_accumulator_t &accumulator, const cgbn_t &value) const;
  __host__ void       sub(cgbn_accumulator_t &accumulator, const cgbn_t &value) const;

  /* math */
  __host__ void       binary_inverse(cgbn_t &r, const cgbn_t &m) const;
  __host__ bool       modular_inverse(cgbn_t &r, const cgbn_t &x, const cgbn_t &modulus) const;
  __host__ void       modular_power(cgbn_t &r, const cgbn_t &a, const cgbn_t &k, const cgbn_t &m) const;
  __host__ void       gcd(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;

  /* fast division: common divisor / modulus */
  __host__ uint32_t   bn2mont(cgbn_t &mont, const cgbn_t &bn, const cgbn_t &n) const;
  __host__ void       mont2bn(cgbn_t &bn, const cgbn_t &mont, const cgbn_t &n, const uint32_t np0) const;
  __host__ void       mont_mul(cgbn_t &r, const cgbn_t &a, const cgbn_t &b, const cgbn_t &n, const uint32_t np0) const;
  __host__ void       mont_sqr(cgbn_t &r, const cgbn_t &a, const cgbn_t &n, const uint32_t np0) const;
  __host__ void       mont_reduce_wide(cgbn_t &r, const cgbn_wide_t &a, const cgbn_t &n, const uint32_t np0) const;

  __host__ uint32_t   barrett_approximation(cgbn_t &approx, const cgbn_t &denom) const;
  __host__ void       barrett_div(cgbn_t &q, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;
  __host__ void       barrett_rem(cgbn_t &r, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;
  __host__ void       barrett_div_rem(cgbn_t &q, cgbn_t &r, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;
  __host__ void       barrett_div_wide(cgbn_t &q, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;
  __host__ void       barrett_rem_wide(cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;
  __host__ void       barrett_div_rem_wide(cgbn_t &q, cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;

  /* load/store to global or shared memory */
  __host__ void       load(cgbn_t &r, cgbn_mem_t<bits> *const address) const;
  __host__ void       store(cgbn_mem_t<bits> *address, const cgbn_t &a) const;

  /* load/store to local memory */
  __host__ void       load(cgbn_t &r, cgbn_local_t *const address) const;
  __host__ void       store(cgbn_local_t *address, const cgbn_t &a) const;
};

#include "impl_mpz.cc"
