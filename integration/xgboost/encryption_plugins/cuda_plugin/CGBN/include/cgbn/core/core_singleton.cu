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

namespace cgbn {

template<class env, uint32_t limbs>
class core_singleton_t;

/* NOTE:  This class name is probably not well named.  
   Singleton connotes a single object instance (similar to a factory object in Java) but that's not what's intended here.
   It's called singleton because it dispatches single limb vs multi-limb cgbn_t's to different APIs in the core_t object. */
   
template<class env>
class core_singleton_t<env, 1> {
  public:
  static const uint32_t limbs=1;
  
  __device__ __forceinline__ static void      mul(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs], const uint32_t add[limbs]) {
    core_t<env>::mul(r[0], a[0], b[0], add[0]);
  }
  
  __device__ __forceinline__ static void      mul_high(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs], const uint32_t add[limbs]) {
    uint32_t ignore;
    
    core_t<env>::mul_wide(ignore, r[0], a[0], b[0], add[0]);
  }

  __device__ __forceinline__ static void      sqrt_resolve_rem(uint32_t rem[limbs], const uint32_t s[limbs], const uint32_t top, const uint32_t r[limbs], const uint32_t shift) {
    core_t<env>::sqrt_resolve_rem(rem[0], s[0], top, r[0], shift);
  }

  __device__ __forceinline__ static void      sqrt(uint32_t s[limbs], const uint32_t a[limbs], const uint32_t numthreads) {
    core_t<env>::sqrt(s[0], a[0], numthreads);
  }
  
  __device__ __forceinline__ static void      sqrt_rem(uint32_t s[limbs], uint32_t r[limbs], const uint32_t a[limbs], const uint32_t numthreads) {
    core_t<env>::sqrt_rem(s[0], r[0], a[0], numthreads);
  }
  
  __device__ __forceinline__ static void      mul_wide(uint32_t lo[limbs], uint32_t hi[limbs], const uint32_t a[limbs], const uint32_t b[limbs], const uint32_t add[limbs]) {
    core_t<env>::mul_wide(lo[0], hi[0], a[0], b[0], add[0]);
  }
  
  __device__ __forceinline__ static void      div_wide(uint32_t q[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t denom[limbs], const uint32_t numthreads) {
    core_t<env>::div_wide(q[0], lo[0], hi[0], denom[0], numthreads);
  }

  __device__ __forceinline__ static void      rem_wide(uint32_t r[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t denom[limbs], const uint32_t numthreads) {
    core_t<env>::rem_wide(r[0], lo[0], hi[0], denom[0], numthreads);
  }
  
  __device__ __forceinline__ static void      div_rem_wide(uint32_t q[limbs], uint32_t r[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t denom[limbs], const uint32_t numthreads) {
    core_t<env>::div_rem_wide(q[0], r[0], lo[0], hi[0], denom[0], numthreads);
  }
  
  __device__ __forceinline__ static void      sqrt_wide(uint32_t s[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t numthreads) {
    core_t<env>::sqrt_wide(s[0], lo[0], hi[0], numthreads);
  }
  
  __device__ __forceinline__ static uint32_t  sqrt_rem_wide(uint32_t s[limbs], uint32_t r[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t numthreads) {
    return core_t<env>::sqrt_rem_wide(s[0], r[0], lo[0], hi[0], numthreads);
  }

  __device__ __forceinline__ static uint32_t  div_ui32(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t div) {
    return core_t<env>::div_ui32(r[0], a[0], div);
  }

  __device__ __forceinline__ static uint32_t  rem_ui32(const uint32_t a[limbs], const uint32_t div) {
    return core_t<env>::rem_ui32(a[0], div);
  }
  
  __device__ __forceinline__ static void      mont_mul(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs], const uint32_t n[limbs], const uint32_t np0) {
    core_t<env>::mont_mul(r[0], a[0], b[0], n[0], np0);
  }

  __device__ __forceinline__ static void      mont_reduce_wide(uint32_t r[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t n[limbs], const uint32_t np0, const bool zero) {
    core_t<env>::mont_reduce_wide(r[0], lo[0], hi[0], n[0], np0, zero);
  }
};

template<class env, uint32_t limbs>
class core_singleton_t {
  public:

  __device__ __forceinline__ static void      mul(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs], const uint32_t add[limbs]) {
    core_t<env>::mul(r, a, b, add);
  }

  __device__ __forceinline__ static void      mul_high(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs], const uint32_t add[limbs]) {
    uint32_t ignore[limbs];
    
    core_t<env>::mul_wide(ignore, r, a, b, add);
  }

  __device__ __forceinline__ static void      sqrt_resolve_rem(uint32_t rem[limbs], const uint32_t s[limbs], const uint32_t top, const uint32_t r[limbs], const uint32_t shift) {
    core_t<env>::sqrt_resolve_rem(rem, s, top, r, shift);
  }
  
  __device__ __forceinline__ static void      sqrt(uint32_t s[limbs], const uint32_t a[limbs], const uint32_t numthreads) {
    core_t<env>::sqrt(s, a, numthreads);
  }

  __device__ __forceinline__ static void      sqrt_rem(uint32_t s[limbs], uint32_t r[limbs], const uint32_t a[limbs], const uint32_t numthreads) {
    core_t<env>::sqrt_rem(s, r, a, numthreads);
  }

  __device__ __forceinline__ static void      mul_wide(uint32_t lo[limbs], uint32_t hi[limbs], const uint32_t a[limbs], const uint32_t b[limbs], const uint32_t add[limbs]) {
    core_t<env>::mul_wide(lo, hi, a, b, add);
  }

  __device__ __forceinline__ static void      div_wide(uint32_t q[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t denom[limbs], const uint32_t numthreads) {
    core_t<env>::div_wide(q, lo, hi, denom, numthreads);
  }
  
  __device__ __forceinline__ static void      rem_wide(uint32_t r[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t denom[limbs], const uint32_t numthreads) {
    core_t<env>::rem_wide(r, lo, hi, denom, numthreads);
  }

  __device__ __forceinline__ static void      div_rem_wide(uint32_t q[limbs], uint32_t r[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t denom[limbs], const uint32_t numthreads) {
    core_t<env>::div_rem_wide(q, r, lo, hi, denom, numthreads);
  }
  
  __device__ __forceinline__ static void      sqrt_wide(uint32_t s[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t numthreads) {
    core_t<env>::sqrt_wide(s, lo, hi, numthreads);
  }
  
  __device__ __forceinline__ static uint32_t  sqrt_rem_wide(uint32_t s[limbs], uint32_t r[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t numthreads) {
    return core_t<env>::sqrt_rem_wide(s, r, lo, hi, numthreads);
  }

  __device__ __forceinline__ static uint32_t  div_ui32(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t div) {
    return core_t<env>::div_ui32(r, a, div);
  }

  __device__ __forceinline__ static uint32_t  rem_ui32(const uint32_t a[limbs], const uint32_t div) {
    return core_t<env>::rem_ui32(a, div);
  }
  
  __device__ __forceinline__ static void      mont_mul(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs], const uint32_t n[limbs], const uint32_t np0) {
    core_t<env>::mont_mul(r, a, b, n, np0);
  }
  
  __device__ __forceinline__ static void      mont_reduce_wide(uint32_t r[limbs], const uint32_t lo[limbs], const uint32_t hi[limbs], const uint32_t n[limbs], const uint32_t np0, const bool zero) {
    core_t<env>::mont_reduce_wide(r, lo, hi, n, np0, zero);
  }
};

} /* namespace cgbn */
