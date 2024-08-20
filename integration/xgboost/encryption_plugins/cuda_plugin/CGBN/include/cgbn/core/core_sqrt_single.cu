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

template<class env>
__device__ __forceinline__ void core_t<env>::sqrt_resolve_rem(uint32_t &rem, const uint32_t s, const uint32_t top, const uint32_t r, const uint32_t shift) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t mask, hi, lo, t, s0;
  
  // remainder computation: r'=(2*s*(s mod b)+r)/b^2
  // where s=square root, r=remainder, b=2^shift.

  hi=(group_thread==TPI-1) ? top : 0;
  
  mask=(1<<shift)-1;
  s0=__shfl_sync(sync, s, 0, TPI) & mask;
  s0=s0+s0;
  lo=madlo_cc(s0, s, r);
  hi=madhic(s0, s, hi);
  hi=resolve_add(hi, lo);

  if(shift>=16) {
    t=__shfl_down_sync(sync, lo, 1, TPI);
    lo=(group_thread==TPI-1) ? hi : t;
    hi=0;
  }
  t=__shfl_down_sync(sync, lo, 1, TPI);
  hi=(group_thread==TPI-1) ? hi : t;
  rem=uright_wrap(lo, hi, shift+shift);
}

template<class env>
__device__ __forceinline__ void core_t<env>::sqrt(uint32_t &s, const uint32_t a, const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t x, x0, x1, t0, t1, divisor, approx, p, q, c, sqrt;

  x=a;
  x0=__shfl_sync(sync, x, TPI-2, TPI);
  x1=__shfl_sync(sync, x, TPI-1, TPI);
  
  divisor=usqrt(x0, x1);
  approx=uapprox(divisor);

  t0=madlo(divisor, divisor, 0);
  t1=madhi(divisor, divisor, 0);
  x0=sub_cc(x0, t0);
  x1=subc(x1, t1);

  x=__shfl_up_sync(sync, x, 1, TPI);
  x=(group_thread==TPI-1) ? x0 : x;
  sqrt=(group_thread==TPI-1) ? divisor+divisor : 0;  // silent 1 at the top of sqrt

  #pragma nounroll
  for(int32_t index=TPI-2;index>=(int32_t)(TPI-numthreads);index--) {
    x0=__shfl_sync(sync, x, TPI-1, TPI);
    q=usqrt_div(x0, x1, divisor, approx);
    sqrt=(group_thread==index) ? q : sqrt;

    p=madhi(q, sqrt, 0);
    x=sub_cc(x, p);
    c=subc(0, 0);
    fast_propagate_sub(c, x);
    
    x1=__shfl_sync(sync, x, TPI-1, TPI)-q;  // we subtract q because of the silent 1 at the top of sqrt
    x=__shfl_up_sync(sync, x, 1, TPI);
    p=madlo(q, sqrt, 0);
    x=sub_cc(x, p);
    c=subc(0, 0);
    x1-=fast_propagate_sub(c, x);
    
    while(0>(int32_t)x1) {
      x1++;
      q--;
      
      // correction step: add q and s
      x=add_cc(x, (group_thread==index) ? q : 0);
      c=addc(0, 0);
      x=add_cc(x, sqrt);
      c=addc(c, 0);
      
      x1+=resolve_add(c, x);

      // update s
      sqrt=(group_thread==index) ? q : sqrt;
    }
    sqrt=(group_thread==index+1) ? sqrt+(q>>31) : sqrt;
    sqrt=(group_thread==index) ? q+q : sqrt;
  }
  c=__shfl_down_sync(sync, sqrt, 1, TPI);
  c=(group_thread==TPI-1) ? 1 : c;
  sqrt=uright_wrap(sqrt, c, 1);
  s=__shfl_sync(sync, sqrt, threadIdx.x-numthreads, TPI);
}

template<class env>
__device__ __forceinline__ void core_t<env>::sqrt_rem(uint32_t &s, uint32_t &r, const uint32_t a, const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t x0, x1, lo, hi, divisor, approx, p, q, c;
  
  r=a;
  x1=__shfl_sync(sync, r, TPI-1, TPI);
  x0=__shfl_sync(sync, r, TPI-2, TPI);
  
  divisor=usqrt(x0, x1);
  approx=uapprox(divisor);

  lo=madlo(divisor, divisor, 0);
  hi=madhi(divisor, divisor, 0);
  lo=sub_cc(x0, lo);
  hi=subc(x1, hi);

  r=__shfl_up_sync(sync, r, 1, TPI);
  r=(group_thread==TPI-1) ? lo : r;
  s=(group_thread==TPI-1) ? divisor+divisor : 0;  // silent 1 at the top of s

  #pragma nounroll
  for(int32_t index=TPI-2;index>=(int32_t)(TPI-numthreads);index--) {
    lo=__shfl_sync(sync, r, TPI-1, TPI);
    q=usqrt_div(lo, hi, divisor, approx);
    s=(group_thread==index) ? q : s;

    p=madhi(q, s, 0);
    r=sub_cc(r, p);
    c=subc(0, 0);
    fast_propagate_sub(c, r);
    
    hi=__shfl_sync(sync, r, TPI-1, TPI)-q;  // we subtract q because of the silent 1 at the top of s
    r=__shfl_up_sync(sync, r, 1, TPI);
    p=madlo(q, s, 0);
    r=sub_cc(r, p);
    c=subc(0, 0);
    hi-=fast_propagate_sub(c, r);
    
    while(0>(int32_t)hi) {
      hi++;
      q--;
      
      // correction step: add q and s
      r=add_cc(r, (group_thread==index) ? q : 0);
      c=addc(0, 0);
      r=add_cc(r, s);
      c=addc(c, 0);
      
      hi+=resolve_add(c, r);

      // update s
      s=(group_thread==index) ? q : s;
    }
    
    s=(group_thread==index+1) ? s + (q>>31) : s;
    s=(group_thread==index) ? q+q : s;
  }
  c=__shfl_down_sync(sync, s, 1, TPI);
  c=(group_thread==TPI-1) ? 1 : c;
  s=uright_wrap(s, c, 1);
  s=__shfl_sync(sync, s, threadIdx.x-numthreads, TPI);

  r=__shfl_sync(sync, r, threadIdx.x-numthreads, TPI);
  r=(group_thread==numthreads) ? hi : r;
  r=(group_thread>numthreads) ? 0 : r;
}

template<class env>
__device__ __forceinline__ void core_t<env>::sqrt_wide(uint32_t &s, const uint32_t lo, const uint32_t hi, const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t x, x0, x1, t0, t1, divisor, approx, p, q, c, sqrt;
  
  x=hi;
  x0=__shfl_sync(sync, x, TPI-2, TPI);
  x1=__shfl_sync(sync, x, TPI-1, TPI);
  
  divisor=usqrt(x0, x1);
  approx=uapprox(divisor);

  t0=madlo_cc(divisor, divisor, 0);
  t1=madhic(divisor, divisor, 0);
  x0=sub_cc(x0, t0);
  x1=subc(x1, t1);

  x=(group_thread==TPI-1) ? lo : x;
  x=__shfl_sync(sync, x, threadIdx.x-1, TPI);
  x=(group_thread==TPI-1) ? x0 : x;
  sqrt=(group_thread==TPI-1) ? divisor+divisor : 0;  // silent 1 at the top of s

  #pragma nounroll
  for(int32_t index=TPI-2;index>=(int32_t)(TPI-numthreads);index--) {
    x0=__shfl_sync(sync, x, TPI-1, TPI);
    q=usqrt_div(x0, x1, divisor, approx);
    sqrt=(group_thread==index) ? q : sqrt;

    p=madhi(q, sqrt, 0);
    x=sub_cc(x, p);
    c=subc(0, 0);
    fast_propagate_sub(c, x);
    
    x1=__shfl_sync(sync, x, TPI-1, TPI)-q;  // we subtract q because of the silent 1 at the top of s
    t0=__shfl_sync(sync, lo, index, TPI);
    x=__shfl_up_sync(sync, x, 1, TPI);
    x=(group_thread==0) ? t0 : x;

    p=madlo(q, sqrt, 0);
    x=sub_cc(x, p);
    c=subc(0, 0);
    x1-=fast_propagate_sub(c, x);
    
    while(0>(int32_t)x1) {
      x1++;
      q--;
      // correction step: add q and sqrt
      x=add_cc(x, (group_thread==index) ? q : 0);
      c=addc(0, 0);
      x=add_cc(x, sqrt);
      c=addc(c, 0);
      
      x1+=resolve_add(c, x);

      // update sqrt
      sqrt=(group_thread==index) ? q : sqrt;
    }
  
    sqrt=(group_thread==index+1) ? sqrt+(q>>31) : sqrt;
    sqrt=(group_thread==index) ? q+q : sqrt;
  }
  c=__shfl_down_sync(sync, sqrt, 1, TPI);
  c=(group_thread==TPI-1) ? 1 : c;
  sqrt=uright_wrap(sqrt, c, 1);
  s=__shfl_sync(sync, sqrt, threadIdx.x-numthreads, TPI);
}

template<class env>
__device__ __forceinline__ uint32_t core_t<env>::sqrt_rem_wide(uint32_t &s, uint32_t &r, const uint32_t lo, const uint32_t hi, const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t x0, x1, t0, t1, divisor, approx, p, q, c, low;
  
  low=lo;
  r=hi;
  x0=__shfl_sync(sync, r, TPI-2, TPI);
  x1=__shfl_sync(sync, r, TPI-1, TPI);
  
  divisor=usqrt(x0, x1);
  approx=uapprox(divisor);

  t0=madlo(divisor, divisor, 0);
  t1=madhi(divisor, divisor, 0);
  x0=sub_cc(x0, t0);
  x1=subc(x1, t1);

  r=(group_thread==TPI-1) ? low : r;
  r=__shfl_sync(sync, r, threadIdx.x-1, TPI);
  r=(group_thread==TPI-1) ? x0 : r;
  s=(group_thread==TPI-1) ? divisor+divisor : 0;  // silent 1 at the top of s

  #pragma nounroll
  for(int32_t index=TPI-2;index>=(int32_t)(TPI-numthreads);index--) {
    x0=__shfl_sync(sync, r, TPI-1, TPI);
    q=usqrt_div(x0, x1, divisor, approx);
    s=(group_thread==index) ? q : s;

    p=madhi(q, s, 0);
    r=sub_cc(r, p);
    c=subc(0, 0);
    fast_propagate_sub(c, r);
    
    x1=__shfl_sync(sync, r, TPI-1, TPI)-q;  // we subtract q because of the silent 1 at the top of s
    t0=__shfl_sync(sync, low, index, TPI);
    r=__shfl_up_sync(sync, r, 1, TPI);
    r=(group_thread==0) ? t0 : r;

    p=madlo(q, s, 0);
    r=sub_cc(r, p);
    c=subc(0, 0);
    x1-=fast_propagate_sub(c, r);
    
    while(0>(int32_t)x1) {
      x1++;
      q--;
      
      // correction step: add q and s
      r=add_cc(r, (group_thread==index) ? q : 0);
      c=addc(0, 0);
      r=add_cc(r, s);
      c=addc(c, 0);
      
      x1+=resolve_add(c, r);

      // update s
      s=(group_thread==index) ? q : s;
    }
    
    s=(group_thread==index+1) ? s+(q>>31) : s;
    s=(group_thread==index) ? q+q : s;
  }
  c=__shfl_down_sync(sync, s, 1, TPI);
  c=(group_thread==TPI-1) ? 1 : c;
  s=uright_wrap(s, c, 1);
  s=__shfl_sync(sync, s, threadIdx.x-numthreads, TPI);

  r=__shfl_sync(sync, r, threadIdx.x-numthreads, TPI);
  r=(group_thread==numthreads) ? x1 : r;
  r=(group_thread>numthreads) ? 0 : r;
  return numthreads==TPI ? x1 : 0;
}

} /* namespace cgbn */