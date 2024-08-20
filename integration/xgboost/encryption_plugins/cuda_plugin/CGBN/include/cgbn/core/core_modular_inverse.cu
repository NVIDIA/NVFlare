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

template<uint32_t tpi, uint32_t limbs>
__device__ __forceinline__ uint32_t top32(const uint32_t sync, uint32_t x[limbs]) {
  return __shfl_sync(sync, x[limbs-1], tpi-1, tpi);
}

template<uint32_t tpi, uint32_t limbs>
__device__ __forceinline__ uint64_t top64(const uint32_t sync, uint32_t x[limbs]) {
  uint32_t lo, hi;

  if(limbs==1)
    lo=__shfl_sync(sync, x[limbs-1], tpi-2, tpi);
  else
    lo=__shfl_sync(sync, x[limbs-2], tpi-1, tpi);
  hi=__shfl_sync(sync, x[limbs-1], tpi-1, tpi);
  return make_wide(lo, hi);
}

template<class env>
__device__ __forceinline__ void core_t<env>::modinv_update_uw_qs(const uint32_t sync, uint32_t r[LIMBS], const uint32_t q, const int32_t s, const uint32_t x[LIMBS]) {
  uint32_t group_thread=threadIdx.x & TPI-1;
  uint32_t temp[LIMBS], lo, hi, top, c;
  
  top=madhi(q, x[LIMBS-1], 0);
  hi=__shfl_sync(sync, top, threadIdx.x-1, TPI);
  if(group_thread==0)
    hi=0;
    
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    lo=madlo_cc(q, x[index], hi);
    hi=madhic(q, x[index], 0);
    temp[index]=lo;
  }
  c=hi-top;
  fast_propagate_add(c, temp);

  rotate_left(temp, temp, s);
  c=mpadd<LIMBS>(r, r, temp);
  fast_propagate_add(c, r);
}

template<class env>
__device__ __forceinline__ void core_t<env>::modinv_update_ab_sq(const uint32_t sync, uint32_t r[LIMBS], const int32_t shift, const uint32_t q, const uint32_t x[LIMBS]) {
  uint32_t group_thread=threadIdx.x & TPI-1;
  uint32_t temp[LIMBS], lo, hi, t, c;
  
  t=__shfl_sync(sync, x[0], threadIdx.x+1, TPI);
  mpright<LIMBS>(temp, x, shift, t);
  
  c=madhi(q, temp[LIMBS-1], 0);
  hi=__shfl_sync(sync, c, threadIdx.x-1, TPI);
  if(group_thread==0)
    hi=0;
    
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    lo=madlo_cc(q, temp[index], hi);
    hi=madhic(q, temp[index], 0);
    temp[index]=lo;
  }
  c=hi-c;
  fast_propagate_add(c, temp);
  
  c=mpsub<LIMBS>(r, r, temp);
  fast_propagate_sub(c, r);
}

template<class env>
__device__ __forceinline__ void core_t<env>::modinv_update_uw(uint32_t sync, uint32_t u[LIMBS], uint32_t w[LIMBS], const unsigned_coeff_t &coeffs) {
  uint32_t group_thread=threadIdx.x & TPI-1;
  uint32_t temp1[LIMBS], temp2[LIMBS], hi, lo, c, top;
  
  // u * alpha_a
  top=madhi(coeffs.alpha_a, u[LIMBS-1], 0);
  hi=__shfl_sync(sync, top, threadIdx.x-1, TPI);
  if(group_thread==0)
    hi=0;
 
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    lo=madlo_cc(coeffs.alpha_a, u[index], hi);
    hi=madhic(coeffs.alpha_a, u[index], 0);
    temp1[index]=lo;
  }
  c=hi-top;

  // w * alpha_a
  top=madhi(coeffs.alpha_b, w[LIMBS-1], 0);
  hi=__shfl_sync(sync, top, threadIdx.x-1, TPI);
  if(group_thread==0)
    hi=0;
 
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    lo=madlo_cc(coeffs.alpha_b, w[index], hi);
    hi=madhic(coeffs.alpha_b, w[index], 0);
    temp2[index]=lo;
  }
  c=c+hi-top+mpadd<LIMBS>(temp1, temp1, temp2);
  resolve_add(c, temp1);
  
  // u * beta_a
  top=madhi(coeffs.beta_a, u[LIMBS-1], 0);
  hi=__shfl_sync(sync, top, threadIdx.x-1, TPI);
  if(group_thread==0)
    hi=0;
 
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    lo=madlo_cc(coeffs.beta_a, u[index], hi);
    hi=madhic(coeffs.beta_a, u[index], 0);
    u[index]=lo;
  }
  c=hi-top;

  // w * beta_b
  top=madhi(coeffs.beta_b, w[LIMBS-1], 0);
  hi=__shfl_sync(sync, top, threadIdx.x-1, TPI);
  if(group_thread==0)
    hi=0;
 
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    lo=madlo_cc(coeffs.beta_b, w[index], hi);
    hi=madhic(coeffs.beta_b, w[index], 0);
    w[index]=lo;
  }
  c=c+hi-top+mpadd<LIMBS>(w, u, w);
  resolve_add(c, w);

  mpset<LIMBS>(u, temp1);
}

template<class env>
__device__ __forceinline__ bool core_t<env>::modinv_small_delta(const uint32_t sync, uint32_t a[LIMBS], uint32_t b[LIMBS], uint32_t u[LIMBS], uint32_t w[LIMBS], int32_t &delta) {
  uint32_t         group_thread=threadIdx.x & TPI-1;
  uint32_t         t;
  int32_t          c, cmp;
  unsigned_coeff_t coeffs;
  
  while(true) {
    coeffs.alpha_a=1; coeffs.alpha_b=0; coeffs.beta_a=0; coeffs.beta_b=1;
    while(true) {
      cmp=ucmp(a[LIMBS-1], b[LIMBS-1]);
      cmp=__shfl_sync(sync, cmp, TPI-1, TPI);
      if(cmp==0) {
        cmp=compare(a, b);
        if(cmp==0) {
          if(delta>0) {
            coeffs.alpha_a=coeffs.alpha_a + ((1<<delta)-1)*coeffs.beta_a;
            coeffs.alpha_b=coeffs.alpha_b + ((1<<delta)-1)*coeffs.beta_b;
          }
          break;
        }
      }
      if(delta>0 || (delta==0 && cmp==1)) {
        if(cmp==-1) {
          c=mpadd<LIMBS>(a, a, a);
          c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
          if(group_thread!=0)
            a[0]=a[0]+c;
          delta--;
        }
        coeffs.alpha_a=coeffs.alpha_a + (coeffs.beta_a<<delta);
        coeffs.alpha_b=coeffs.alpha_b + (coeffs.beta_b<<delta);
        c=mpsub<LIMBS>(a, a, b);
        fast_propagate_sub(c, a);
        t=__shfl_sync(sync, a[LIMBS-1], TPI-1, TPI);
        if(t!=0) {
          c=ushiftamt(t);
          delta-=c;
          t=__shfl_sync(sync, a[LIMBS-1], threadIdx.x-1, TPI);
          mpleft<LIMBS>(a, a, c, t);
        }
        else {
          c=clz(a);
          delta-=c;
          rotate_left(a, a, c);
        }
      }
      else {
        if(cmp==1) {
          c=mpadd<LIMBS>(b, b, b);
          c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
          if(group_thread!=0)
            b[0]=b[0]+c;
          delta++;
        }
        coeffs.beta_a=coeffs.beta_a + (coeffs.alpha_a<<-delta);
        coeffs.beta_b=coeffs.beta_b + (coeffs.alpha_b<<-delta);
        c=mpsub<LIMBS>(b, b, a);
        fast_propagate_sub(c, b);
        t=__shfl_sync(sync, b[LIMBS-1], TPI-1, TPI);
        if(t!=0) {
          c=ushiftamt(t);
          delta+=c;
          t=__shfl_sync(sync, b[LIMBS-1], threadIdx.x-1, TPI);
          mpleft<LIMBS>(b, b, c, t);
        }
        else {
          c=clz(b);
          delta+=c;
          rotate_left(b, b, c);
        }
      }
      if(uabs(delta)>=8) {
        // faster to do a divide
        break;
      }
      if((coeffs.alpha_a | coeffs.alpha_b | coeffs.beta_a | coeffs.beta_b)>0xFFFFFF) {
        // poor mans max
        break;
      }
    }

    // we have accumulated a bunch of operations, apply them to u and w
    modinv_update_uw(sync, u, w, coeffs);
    if(cmp==0)
      return true;
    if(uabs(delta)>=8)
      return false;
  }
}

template<class env>
__device__ __forceinline__ bool core_t<env>::modular_inverse(uint32_t inv[LIMBS], const uint32_t x[LIMBS], const uint32_t y[LIMBS]) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1, warp_thread=threadIdx.x & warpSize-1;
  uint32_t a[LIMBS], b[LIMBS], u[LIMBS], w[LIMBS], d_top, q, ballot;
  uint64_t n_top;
  int32_t  delta, ad, c;

  if(__shfl_sync(sync, (x[0] | y[0]) & 0x01, 0, TPI)==0) {
    mpzero<LIMBS>(inv);
    return false;
  }
  
  mpset<LIMBS>(a, x);
  mpset<LIMBS>(b, y);
  mpzero<LIMBS>(u);
  mpzero<LIMBS>(w);
  if(group_thread==0)
    u[0]=1;
  
  c=clz(a);
  delta=clz(b);
  if(c==BITS || delta>=BITS-1) {
    mpzero<LIMBS>(inv);
    return false;
  }
  rotate_left(a, a, c);
  rotate_left(b, b, delta);
  delta=delta-c;
  
  while(true) {
    ad=uabs(delta);
/*
if(blockIdx.x==0 && threadIdx.x==0) printf("delta=%d\n", ad);
if(blockIdx.x==0 && threadIdx.x<32) {
  for(int32_t index=31;index>=0;index--) {
    uint32_t tt=__shfl_sync(sync, a[0], index, TPI);
    if(threadIdx.x==0) printf("%08X", tt);
  }
  if(threadIdx.x==0) printf("\n");
}
if(blockIdx.x==0 && threadIdx.x<32) {
  for(int32_t index=31;index>=0;index--) {
    uint32_t tt=__shfl_sync(sync, b[0], index, TPI);
    if(threadIdx.x==0) printf("%08X", tt);
  }
  if(threadIdx.x==0) printf("\n");
}
*/
    if(ad<8) {
      if(modinv_small_delta(sync, a, b, u, w, delta))
        break;
    }
    else {
      if(delta>0) {
        n_top=top64<TPI, LIMBS>(sync, a);
        d_top=top32<TPI, LIMBS>(sync, b);
      }
      else {
        n_top=top64<TPI, LIMBS>(sync, b);
        d_top=top32<TPI, LIMBS>(sync, a);
      }
      c=32;
      if(ad<32) {
        c=ad;
        n_top=n_top>>32-c;
      }
      else if(uhigh(n_top)>=d_top) {
        n_top=n_top>>1;
        c--;
      }
      q=n_top/d_top-2;
//if(blockIdx.x==0 && threadIdx.x==0) printf("%016lX %08X   q=%08X\n", n_top, d_top, q);

      if(delta>0) {
        modinv_update_uw_qs(sync, u, q, delta-c, w);
        modinv_update_ab_sq(sync, a, c, q, b);
        c=clz(a);
        if(c==BITS) {
          mpset<LIMBS>(a, b);
          break;
        }
        rotate_left(a, a, c);
        delta-=c;
      }
      else {
        modinv_update_uw_qs(sync, w, q, -delta-c, u);
        modinv_update_ab_sq(sync, b, c, q, a);
        c=clz(b);
        if(c==BITS)
          break;
        rotate_left(b, b, c);
        delta+=c;
      }
    }
  }

  if(group_thread==TPI-1) 
    a[LIMBS-1]=a[LIMBS-1] ^ 0x80000000;
  ballot=__ballot_sync(sync, mplor<LIMBS>(a)==0);
  if(TPI<warpSize)
    ballot=(ballot>>(warp_thread^group_thread)) & TPI_ONES;
  if(ballot!=TPI_ONES) {
    // gcd is not 1
    mpzero<LIMBS>(inv);
    return false;
  }
  
  mpset<LIMBS>(inv, u);
  return true;
}



/*
if(blockIdx.x==0 && threadIdx.x<32) {
  for(int32_t index=31;index>=0;index--) {
    uint32_t tt=__shfl_sync(sync, u[0], index, TPI);
    if(threadIdx.x==0) printf("%08X", tt);
  }
  if(threadIdx.x==0) printf("\n");
}
if(blockIdx.x==0 && threadIdx.x<32) {
  for(int32_t index=31;index>=0;index--) {
    uint32_t tt=__shfl_sync(sync, w[0], index, TPI);
    if(threadIdx.x==0) printf("%08X", tt);
  }
  if(threadIdx.x==0) printf("\n");
}
*/

/*
if(blockIdx.x==0 && threadIdx.x>=32 && threadIdx.x<64) {
  for(int32_t index=31;index>=0;index--) {
    uint32_t tt=__shfl_sync(sync, u[0], index, TPI);
    if(group_thread==0) printf("%08X", tt);
  }
  if(threadIdx.x==0) printf("\n");
}
*/


} /* namespace cgbn */