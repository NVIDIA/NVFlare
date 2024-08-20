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
__device__ __forceinline__ void core_t<env>::div_wide(uint32_t &q, const uint32_t lo, const uint32_t hi, const uint32_t denom, const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t x, d0, d1, x0, x1, x2, est, t, a, h, l, quotient;
  int32_t  c, top;

  quotient=0;
  x=hi;
  if(numthreads<TPI) {
    x=(group_thread<numthreads) ? hi : lo;
    x=__shfl_sync(sync, x, threadIdx.x+numthreads, TPI);
    t=sub_cc(x, denom);
    c=subc(0, 0);
    if(resolve_sub(c, t)==0) {
      x=t;
      quotient=(group_thread==numthreads) ? 1 : quotient;
    }
  }

  d0=__shfl_sync(sync, denom, TPI-2, TPI);
  d1=__shfl_sync(sync, denom, TPI-1, TPI);
    
  a=uapprox(d1);
 
  #pragma nounroll
  for(int32_t thread=numthreads-1;thread>=0;thread--) {
    x0=__shfl_sync(sync, x, TPI-3, TPI);
    x1=__shfl_sync(sync, x, TPI-2, TPI);
    x2=__shfl_sync(sync, x, TPI-1, TPI);
    est=udiv(x0, x1, x2, d0, d1, a);

    t=__shfl_sync(sync, lo, thread, TPI);
    l=madlo(est, denom, 0);
    h=madhi(est, denom, 0);

    x=sub_cc(x, h);
    c=subc(0, 0);  // thread TPI-1 is zero
    
    top=__shfl_sync(sync, x, TPI-1, TPI);
    x=__shfl_sync(sync, x, threadIdx.x-1, TPI);
    c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
    x=(group_thread==0) ? t : x;

    x=sub_cc(x, l);
    c=subc(c, 0);

    if(top+resolve_sub(c, x)<0) {
      // means a correction is required, should be very rare
      x=add_cc(x, denom);
      c=addc(0, 0);
      fast_propagate_add(c, x);
      est--;
    }
    quotient=(group_thread==thread) ? est : quotient;
  }
  q=quotient;
}

template<class env>
__device__ __forceinline__ void core_t<env>::rem_wide(uint32_t &r, const uint32_t lo, const uint32_t hi, const uint32_t denom, const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t x, d0, d1, x0, x1, x2, est, t, a, h, l;
  int32_t  c, top;

  x=hi;
  if(numthreads<TPI) {
    x=(group_thread<numthreads) ? hi : lo;
    x=__shfl_sync(sync, x, threadIdx.x+numthreads, TPI);
    t=sub_cc(x, denom);
    c=subc(0, 0);
    if(resolve_sub(c, t)==0) 
      x=t;
  }

  d0=__shfl_sync(sync, denom, TPI-2, TPI);
  d1=__shfl_sync(sync, denom, TPI-1, TPI);
    
  a=uapprox(d1);

  #pragma nounroll
  for(int32_t thread=numthreads-1;thread>=0;thread--) {
    x0=__shfl_sync(sync, x, TPI-3, TPI);
    x1=__shfl_sync(sync, x, TPI-2, TPI);
    x2=__shfl_sync(sync, x, TPI-1, TPI);
    est=udiv(x0, x1, x2, d0, d1, a);

    t=__shfl_sync(sync, lo, thread, TPI);
    l=madlo(est, denom, 0);
    h=madhi(est, denom, 0);

    x=sub_cc(x, h);
    c=subc(0, 0);  // thread TPI-1 is zero
    
    top=__shfl_sync(sync, x, TPI-1, TPI);
    x=__shfl_sync(sync, x, threadIdx.x-1, TPI);
    c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
    x=(group_thread==0) ? t : x;

    x=sub_cc(x, l);
    c=subc(c, 0);
    
    if(top+resolve_sub(c, x)<0) {
      // means a correction is required, should be very rare
      x=add_cc(x, denom);
      c=addc(0, 0);
      fast_propagate_add(c, x);
    }
  }
  r=x;
}

template<class env>
__device__ __forceinline__ void core_t<env>::div_rem_wide(uint32_t &q, uint32_t &r, const uint32_t lo, const uint32_t hi, const uint32_t denom, const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t x, d0, d1, x0, x1, x2, est, t, a, h, l, quotient;
  int32_t  c, top;

  quotient=0;
  x=hi;
  if(numthreads<TPI) {
    x=(group_thread<numthreads) ? hi : lo;
    x=__shfl_sync(sync, x, threadIdx.x+numthreads, TPI);
    t=sub_cc(x, denom);
    c=subc(0, 0);
    if(resolve_sub(c, t)==0) {
      x=t;
      quotient=(group_thread==numthreads) ? 1 : quotient;
    }
  }

  d0=__shfl_sync(sync, denom, TPI-2, TPI);
  d1=__shfl_sync(sync, denom, TPI-1, TPI);
    
  a=uapprox(d1);

  #pragma nounroll
  for(int32_t thread=numthreads-1;thread>=0;thread--) {
    x0=__shfl_sync(sync, x, TPI-3, TPI);
    x1=__shfl_sync(sync, x, TPI-2, TPI);
    x2=__shfl_sync(sync, x, TPI-1, TPI);
    est=udiv(x0, x1, x2, d0, d1, a);

    t=__shfl_sync(sync, lo, thread, TPI);
    l=madlo(est, denom, 0);
    h=madhi(est, denom, 0);

    x=sub_cc(x, h);
    c=subc(0, 0);  // thread TPI-1 is zero
    
    top=__shfl_sync(sync, x, TPI-1, TPI);
    x=__shfl_sync(sync, x, threadIdx.x-1, TPI);
    c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
    x=(group_thread==0) ? t : x;

    x=sub_cc(x, l);
    c=subc(c, 0);
    
    if(top+resolve_sub(c, x)<0) {
      // means a correction is required, should be very rare
      x=add_cc(x, denom);
      c=addc(0, 0);
      fast_propagate_add(c, x);
      est--;
    }
    quotient=(group_thread==thread) ? est : quotient;
  }
  q=quotient;
  r=x;
}

} /* namespace cgbn */