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

#include <cooperative_groups.h>
namespace cg=cooperative_groups;

typedef enum {
  cgbn_instance_syncable,
  cgbn_warp_syncable,
  cgbn_block_syncable,
  cgbn_grid_syncable
} cgbn_syncable_t;

class cgbn_default_parameters_t {
  public:

  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;
};

/* forward declarations */
template<uint32_t tpi, class params>
class cgbn_context_t;

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
class cgbn_env_t;

template<uint32_t bits>
struct cgbn_mem_t {
  public:
  uint32_t _limbs[(bits+31)/32];
};

/* main classes */
template<uint32_t tpi, class params=cgbn_default_parameters_t>
class cgbn_context_t {
  public:
  static const uint32_t TPB=params::TPB;
  static const uint32_t TPI=tpi;
  static const uint32_t MAX_ROTATION=params::MAX_ROTATION;
  static const uint32_t SHM_LIMIT=params::SHM_LIMIT;
  static const bool     CONSTANT_TIME=params::CONSTANT_TIME;

  const cgbn_monitor_t  _monitor;
  cgbn_error_report_t  *const _report;
  const int32_t         _instance;
  uint32_t             *_scratch;

  public:
  __device__ __forceinline__ cgbn_context_t();
  __device__ __forceinline__ cgbn_context_t(cgbn_monitor_t type);
  __device__ __forceinline__ cgbn_context_t(cgbn_monitor_t type, cgbn_error_report_t *report);
  __device__ __forceinline__ cgbn_context_t(cgbn_monitor_t type, cgbn_error_report_t *report, uint32_t instance);

  __device__ __forceinline__ uint32_t *scratch() const;
  __device__ __forceinline__ bool      check_errors() const;
  __device__ __noinline__    void      report_error(cgbn_error_t error) const;

  template<class env_t>
  __device__ __forceinline__ env_t env() {
    env_t env(*this);

    return env;
  }

  template<uint32_t bits, cgbn_syncable_t syncable>
  __device__ __forceinline__ cgbn_env_t<cgbn_context_t, bits, syncable> env() {
    cgbn_env_t<cgbn_context_t, bits, syncable> env(*this);

    return env;
  }
};

template<class context_t, uint32_t bits, cgbn_syncable_t syncable=cgbn_instance_syncable>
class cgbn_env_t {
  public:

  // bits must be divisible by 32
  static const uint32_t        BITS=bits;
  static const uint32_t        TPB=context_t::TPB;
  static const uint32_t        TPI=context_t::TPI;
  static const uint32_t        MAX_ROTATION=context_t::MAX_ROTATION;
  static const uint32_t        SHM_LIMIT=context_t::SHM_LIMIT;
  static const bool            CONSANT_TIME=context_t::CONSTANT_TIME;
  static const cgbn_syncable_t SYNCABLE=syncable;

  static const uint32_t        LIMBS=(bits/32+TPI-1)/TPI;
  static const uint32_t        LOCAL_LIMBS=((bits+32)/64+TPI-1)/TPI*TPI;
  static const uint32_t        UNPADDED_BITS=TPI*LIMBS*32;
  static const uint32_t        PADDING=bits/32%TPI;
  static const uint32_t        PAD_THREAD=(BITS/32)/LIMBS;
  static const uint32_t        PAD_LIMB=(BITS/32)%LIMBS;

  struct cgbn_t {
    public:
    typedef cgbn_env_t parent_env_t;

    uint32_t _limbs[LIMBS];
  };
  struct cgbn_wide_t {
    public:
    cgbn_t _low, _high;
  };
  struct cgbn_local_t {
    public:
    uint64_t _limbs[LOCAL_LIMBS];
  };
  struct cgbn_accumulator_t {
    public:
    uint32_t _carry;
    uint32_t _limbs[LIMBS];
    __device__ __forceinline__ cgbn_accumulator_t();
  };

  const context_t &_context;

  __device__ __forceinline__ cgbn_env_t(const context_t &context);

  /* size conversion */
  template<typename source_cgbn_t>
  __device__ __forceinline__ void       set(cgbn_t &r, const source_cgbn_t &source) const;

  /* bn arithmetic routines */
  __device__ __forceinline__ void       set(cgbn_t &r, const cgbn_t &a) const;
  __device__ __forceinline__ void       swap(cgbn_t &r, cgbn_t &a) const;
  __device__ __forceinline__ int32_t    add(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __device__ __forceinline__ int32_t    sub(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __device__ __forceinline__ int32_t    negate(cgbn_t &r, const cgbn_t &a) const;
  __device__ __forceinline__ void       mul(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __device__ __forceinline__ void       mul_high(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __device__ __forceinline__ void       sqr(cgbn_t &r, const cgbn_t &a) const;
  __device__ __forceinline__ void       sqr_high(cgbn_t &r, const cgbn_t &a) const;
  __device__ __forceinline__ void       div(cgbn_t &q, const cgbn_t &num, const cgbn_t &denom) const;
  __device__ __forceinline__ void       rem(cgbn_t &r, const cgbn_t &num, const cgbn_t &denom) const;
  __device__ __forceinline__ void       div_rem(cgbn_t &q, cgbn_t &r, const cgbn_t &num, const cgbn_t &denom) const;
  __device__ __forceinline__ void       sqrt(cgbn_t &s, const cgbn_t &a) const;
  __device__ __forceinline__ void       sqrt_rem(cgbn_t &s, cgbn_t &r, const cgbn_t &a) const;
  __device__ __forceinline__ bool       equals(const cgbn_t &a, const cgbn_t &b) const;
  __device__ __forceinline__ int32_t    compare(const cgbn_t &a, const cgbn_t &b) const;
  __device__ __forceinline__ void       extract_bits(cgbn_t &r, const cgbn_t &a, const uint32_t start, const uint32_t len) const;
  __device__ __forceinline__ void       insert_bits(cgbn_t &r, const cgbn_t &a, const uint32_t start, const uint32_t len, const cgbn_t &value) const;

  /* ui32 arithmetic routines*/
  __device__ __forceinline__ uint32_t   get_ui32(const cgbn_t &a) const;
  __device__ __forceinline__ void       set_ui32(cgbn_t &r, const uint32_t value) const;
  __device__ __forceinline__ int32_t    add_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t add) const;
  __device__ __forceinline__ int32_t    sub_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t sub) const;
  __device__ __forceinline__ uint32_t   mul_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t mul) const;
  __device__ __forceinline__ uint32_t   div_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t div) const;
  __device__ __forceinline__ uint32_t   rem_ui32(const cgbn_t &a, const uint32_t div) const;
  __device__ __forceinline__ bool       equals_ui32(const cgbn_t &a, const uint32_t value) const;
  __device__ __forceinline__ int32_t    compare_ui32(const cgbn_t &a, const uint32_t value) const;
  __device__ __forceinline__ uint32_t   extract_bits_ui32(const cgbn_t &a, const uint32_t start, const uint32_t len) const;
  __device__ __forceinline__ void       insert_bits_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t start, const uint32_t len, const uint32_t value) const;
  __device__ __forceinline__ uint32_t   binary_inverse_ui32(const uint32_t n0) const;
  __device__ __forceinline__ uint32_t   gcd_ui32(const cgbn_t &a, const uint32_t value) const;

  /* wide arithmetic routines */
  __device__ __forceinline__ void       mul_wide(cgbn_wide_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __device__ __forceinline__ void       sqr_wide(cgbn_wide_t &r, const cgbn_t &a) const;
  __device__ __forceinline__ void       div_wide(cgbn_t &q, const cgbn_wide_t &num, const cgbn_t &denom) const;
  __device__ __forceinline__ void       rem_wide(cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom) const;
  __device__ __forceinline__ void       div_rem_wide(cgbn_t &q, cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom) const;
  __device__ __forceinline__ void       sqrt_wide(cgbn_t &s, const cgbn_wide_t &a) const;
  __device__ __forceinline__ void       sqrt_rem_wide(cgbn_t &s, cgbn_wide_t &r, const cgbn_wide_t &a) const;

  /* logical, shifting, masking */
  __device__ __forceinline__ void       bitwise_and(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __device__ __forceinline__ void       bitwise_ior(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __device__ __forceinline__ void       bitwise_xor(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __device__ __forceinline__ void       bitwise_complement(cgbn_t &r, const cgbn_t &a) const;
  __device__ __forceinline__ void       bitwise_select(cgbn_t &r, const cgbn_t &clear, const cgbn_t &set, const cgbn_t &select) const;
  __device__ __forceinline__ void       bitwise_mask_copy(cgbn_t &r, const int32_t numbits) const;
  __device__ __forceinline__ void       bitwise_mask_and(cgbn_t &r, const cgbn_t &a, const int32_t numbits) const;
  __device__ __forceinline__ void       bitwise_mask_ior(cgbn_t &r, const cgbn_t &a, const int32_t numbits) const;
  __device__ __forceinline__ void       bitwise_mask_xor(cgbn_t &r, const cgbn_t &a, const int32_t numbits) const;
  __device__ __forceinline__ void       bitwise_mask_select(cgbn_t &r, const cgbn_t &clear, const cgbn_t &set, int32_t numbits) const;
  __device__ __forceinline__ void       shift_left(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const;
  __device__ __forceinline__ void       shift_right(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const;
  __device__ __forceinline__ void       rotate_left(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const;
  __device__ __forceinline__ void       rotate_right(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const;

  /* bit counting */
  __device__ __forceinline__ uint32_t   pop_count(const cgbn_t &a) const;
  __device__ __forceinline__ uint32_t   clz(const cgbn_t &a) const;
  __device__ __forceinline__ uint32_t   ctz(const cgbn_t &a) const;

  /* accumulator APIs */
  __device__ __forceinline__ int32_t    resolve(cgbn_t &sum, const cgbn_accumulator_t &accumulator) const;
  __device__ __forceinline__ void       set(cgbn_accumulator_t &accumulator, const cgbn_t &value) const;
  __device__ __forceinline__ void       add(cgbn_accumulator_t &accumulator, const cgbn_t &value) const;
  __device__ __forceinline__ void       sub(cgbn_accumulator_t &accumulator, const cgbn_t &value) const;
  __device__ __forceinline__ void       set_ui32(cgbn_accumulator_t &accumulator, const uint32_t value) const;
  __device__ __forceinline__ void       add_ui32(cgbn_accumulator_t &accumulator, const uint32_t value) const;
  __device__ __forceinline__ void       sub_ui32(cgbn_accumulator_t &accumulator, const uint32_t value) const;

  /* math */
  __device__ __forceinline__ void       binary_inverse(cgbn_t &r, const cgbn_t &x) const;
  __device__ __forceinline__ void       gcd(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const;
  __device__ __forceinline__ bool       modular_inverse(cgbn_t &r, const cgbn_t &x, const cgbn_t &modulus) const;
  __device__ __forceinline__ void       modular_power(cgbn_t &r, const cgbn_t &x, const cgbn_t &exponent, const cgbn_t &modulus) const;

  /* fast division: common divisor / modulus */
  __device__ __forceinline__ uint32_t   bn2mont(cgbn_t &mont, const cgbn_t &bn, const cgbn_t &n) const;
  __device__ __forceinline__ void       mont2bn(cgbn_t &bn, const cgbn_t &mont, const cgbn_t &n, const uint32_t np0) const;
  __device__ __forceinline__ void       mont_mul(cgbn_t &r, const cgbn_t &a, const cgbn_t &b, const cgbn_t &n, const uint32_t np0) const;
  __device__ __forceinline__ void       mont_sqr(cgbn_t &r, const cgbn_t &a, const cgbn_t &n, const uint32_t np0) const;
  __device__ __forceinline__ void       mont_reduce_wide(cgbn_t &r, const cgbn_wide_t &a, const cgbn_t &n, const uint32_t np0) const;

  __device__ __forceinline__ uint32_t   barrett_approximation(cgbn_t &approx, const cgbn_t &denom) const;
  __device__ __forceinline__ void       barrett_div(cgbn_t &q, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;
  __device__ __forceinline__ void       barrett_rem(cgbn_t &r, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;
  __device__ __forceinline__ void       barrett_div_rem(cgbn_t &q, cgbn_t &r, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;
  __device__ __forceinline__ void       barrett_div_wide(cgbn_t &q, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;
  __device__ __forceinline__ void       barrett_rem_wide(cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;
  __device__ __forceinline__ void       barrett_div_rem_wide(cgbn_t &q, cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const;

  /* load/store to global or shared memory */
  __device__ __forceinline__ void       load(cgbn_t &r, cgbn_mem_t<bits> *const address) const;
  __device__ __forceinline__ void       store(cgbn_mem_t<bits> *address, const cgbn_t &a) const;

  /* load/store to local memory */
  __device__ __forceinline__ void       load(cgbn_t &r, cgbn_local_t *const address) const;
  __device__ __forceinline__ void       store(cgbn_local_t *address, const cgbn_t &a) const;
};

#include "impl_cuda.cu"

/*
experimental:

  // faster shift and rotate by a constant number of bits
  template<uint32_t numbits> __device__ __forceinline__ void shift_left(cgbn_t &r, const cgbn_t &a);
  template<uint32_t numbits> __device__ __forceinline__ void shift_right(cgbn_t &r, const cgbn_t &a);
  template<uint32_t numbits> __device__ __forceinline__ void rotate_left(cgbn_t &r, const cgbn_t &a);
  template<uint32_t numbits> __device__ __forceinline__ void rotate_right(cgbn_t &r, const cgbn_t &a);

*/
