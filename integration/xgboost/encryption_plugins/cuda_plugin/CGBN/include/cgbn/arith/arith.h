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

#include <cstdint>

namespace cgbn {

/* defines */
#ifdef NDEBUG
#define ASM_ERROR(message) asm volatile("ASM_ERROR " message)
#else
#define ASM_ERROR(message)
#endif

/* static math */
template<uint32_t denominator> __device__ __forceinline__ uint32_t static_divide_small(uint32_t numerator);
template<uint32_t denominator> __device__ __forceinline__ uint32_t static_remainder_small(uint32_t numerator);

/* asm routines */
__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b);
__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b);
__device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b);
__device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b);
__device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b);
__device__ __forceinline__ uint32_t subc(uint32_t a, uint32_t b);
__device__ __forceinline__ uint32_t madlo(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t madloc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t madhi(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t madhic(uint32_t a, uint32_t b, uint32_t c);

/* xmad routines */
__device__ __forceinline__ uint32_t xmadll(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadll_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadllc_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadllc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadlh(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadlh_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadlhc_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadlhc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadhl(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadhl_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadhlc_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadhlc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadhh(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadhh_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadhhc_cc(uint32_t a, uint32_t b, uint32_t c);
__device__ __forceinline__ uint32_t xmadhhc(uint32_t a, uint32_t b, uint32_t c);

/* funnel shifts */
__device__ __forceinline__ uint32_t uleft_clamp(uint32_t lo, uint32_t hi, uint32_t amt);
__device__ __forceinline__ uint32_t uright_clamp(uint32_t lo, uint32_t hi, uint32_t amt);

__device__ __forceinline__ uint32_t uleft_wrap(uint32_t lo, uint32_t hi, uint32_t amt);
__device__ __forceinline__ uint32_t uright_wrap(uint32_t lo, uint32_t hi, uint32_t amt);

__device__ __forceinline__ uint32_t uabs(int32_t x);

#define CGBN_INF_CHAIN 0xFFFFFFFF

/* classes */
template<uint32_t length=CGBN_INF_CHAIN, bool carry_in=false, bool carry_out=false>
class chain_t {
  public:
  uint32_t _position;

  __device__ __forceinline__ chain_t();
  __device__ __forceinline__ ~chain_t();
  __device__ __forceinline__ uint32_t add(uint32_t a, uint32_t b);
  __device__ __forceinline__ uint32_t sub(uint32_t a, uint32_t b);
  __device__ __forceinline__ uint32_t madlo(uint32_t a, uint32_t b, uint32_t c);
  __device__ __forceinline__ uint32_t madhi(uint32_t a, uint32_t b, uint32_t c);
  __device__ __forceinline__ uint32_t xmadll(uint32_t a, uint32_t b, uint32_t c);
  __device__ __forceinline__ uint32_t xmadlh(uint32_t a, uint32_t b, uint32_t c);
  __device__ __forceinline__ uint32_t xmadhl(uint32_t a, uint32_t b, uint32_t c);
  __device__ __forceinline__ uint32_t xmadhh(uint32_t a, uint32_t b, uint32_t c);
};


/* uint32 math routines */
__device__ __forceinline__ uint32_t uclz(uint32_t x);
__device__ __forceinline__ uint32_t uctz(uint32_t x);
__device__ __forceinline__ uint32_t ubinary_inverse(uint32_t x);
__device__ __forceinline__ uint32_t ugcd(uint32_t a, uint32_t b);
__device__ __forceinline__ uint32_t uapprox(const uint32_t d);
__device__ __forceinline__ uint32_t udiv(const uint32_t lo, const uint32_t hi, const uint32_t d, const uint32_t approx);
__device__ __forceinline__ uint32_t udiv(const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t d0, const uint32_t d1, const uint32_t approx);


/* mp math routines */
template<uint32_t limbs> __device__ __forceinline__ uint32_t mplor(const uint32_t a[limbs]);
template<uint32_t limbs> __device__ __forceinline__ uint32_t mpland(const uint32_t a[limbs]);
template<uint32_t limbs> __device__ __forceinline__ bool     mpzeros(const uint32_t a[limbs]);
template<uint32_t limbs> __device__ __forceinline__ bool     mpones(const uint32_t a[limbs]);

template<uint32_t limbs> __device__ __forceinline__ void     mpadd32_cc(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b);
template<uint32_t limbs> __device__ __forceinline__ uint32_t mpadd32(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b);
template<uint32_t limbs> __device__ __forceinline__ void     mpadd_cc(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs]);
template<uint32_t limbs> __device__ __forceinline__ uint32_t mpadd(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs]);

template<uint32_t limbs> __device__ __forceinline__ void     mpsub32_cc(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b);
template<uint32_t limbs> __device__ __forceinline__ uint32_t mpsub32(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b);
template<uint32_t limbs> __device__ __forceinline__ void     mpsub_cc(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs]);
template<uint32_t limbs> __device__ __forceinline__ uint32_t mpsub(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs]);

template<uint32_t limbs> __device__ __forceinline__ uint32_t mpmul32(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b);
template<uint32_t limbs> __device__ __forceinline__ uint32_t mprem32(const uint32_t a[limbs], const uint32_t d, const uint32_t approx);

template<uint32_t limbs> __device__ __forceinline__ void     mpmul(uint32_t lo[limbs], uint32_t hi[limbs], const uint32_t a[limbs], const uint32_t b[limbs]);

template<uint32_t limbs, uint32_t max_rotation> __device__ __forceinline__ void mprotate_left(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t numlimbs);
template<uint32_t limbs, uint32_t max_rotation> __device__ __forceinline__ void mprotate_right(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t numlimbs);

}  /* CGBN namespace */

#include "static_divide.cu"
#include "asm.cu"
#include "chain.cu"
#include "math.cu"
#include "shifter.cu"
#include "mp.cu"
#include "dmp.cu"
