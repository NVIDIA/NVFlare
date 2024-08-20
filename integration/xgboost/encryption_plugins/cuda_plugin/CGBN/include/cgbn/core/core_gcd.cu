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

__device__ __forceinline__ static void gcd_reduce(signed_coeff_t &coeffs, int32_t a, int32_t b) {
  int32_t  c0, c1, t, q, i, n=b-a, count=15;
  
  coeffs.alpha_a=1;
  coeffs.alpha_b=0;
  coeffs.beta_a=-1;
  coeffs.beta_b=1;
  
  // a and b must both be odd
  c1=0;
  while(1) {
    t=n^n-1;
    c0=31-ushiftamt(t);
    count=count-c0;
    if(count<0) {
      coeffs.alpha_a=coeffs.alpha_a<<c1;
      coeffs.alpha_b=coeffs.alpha_b<<c1;
      break;
    }
    
    b=n>>c0;
    i=b*(b*b+62);   // 6 bit inverse
    if(t>0x3F) {
      i=i*(i*b+2);  // 12 bit inverse
      i=i*(i*b+2);  // 24 bit inverse
    }

    q=a*i & t;
    t=q & n;
    q=q-t-t;
    n=b*q+a>>c0;

    coeffs.alpha_a=(coeffs.alpha_a<<c0+c1) + q*coeffs.beta_a;
    coeffs.alpha_b=(coeffs.alpha_b<<c0+c1) + q*coeffs.beta_b;

    t=n^n-1;
    c1=31-ushiftamt(t);
    count=count-c1;
    if(count<0) {
      coeffs.beta_a=coeffs.beta_a<<c0;
      coeffs.beta_b=coeffs.beta_b<<c0;
      break;
    }
    
    a=n>>c1;
    i=a*(a*a+62);   // 6 bit inverse
    if(t>0x3F) {
      i=i*(i*a+2);  // 12 bit inverse
      i=i*(i*a+2);  // 24 bit inverse
    }

    q=b*i & t;
    t=q & n;
    q=q-t-t;
    n=a*q+b>>c1;    

    coeffs.beta_a=(coeffs.beta_a<<c1+c0) + q*coeffs.alpha_a;
    coeffs.beta_b=(coeffs.beta_b<<c1+c0) + q*coeffs.alpha_b;
  }
}

template<class env>
__device__ __forceinline__ void core_t<env>::gcd_product(const uint32_t sync, uint32_t r[LIMBS], const int32_t sa, const uint32_t a[LIMBS], const int32_t sb, const uint32_t b[LIMBS]) {
  uint32_t group_thread=threadIdx.x & TPI-1;
  uint32_t c, t, ua, ub, a_high, b_high;
  int32_t  sign;
  
  a_high=__shfl_sync(sync, a[LIMBS-1], group_thread-1, TPI);
  b_high=__shfl_sync(sync, b[LIMBS-1], group_thread-1, TPI);

  sign=sa ^ sb;
  ua=uabs(sa);
  ub=uabs(sb);  

  if(sign>=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      r[index]=madlo(ua, a[index], 0);

    chain_t<> chain1;
    r[0]=chain1.madhi(ua, a_high, r[0]);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      r[index]=chain1.madhi(ua, a[index-1], r[index]);
    c=chain1.add(0, 0);

    chain_t<> chain2;
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      r[index]=chain2.madlo(ub, b[index], r[index]);
    c=chain2.add(c, 0);
    
    chain_t<> chain3;
    r[0]=chain3.madhi(ub, b_high, r[0]);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++) 
      r[index]=chain3.madhi(ub, b[index-1], r[index]);
    c=chain3.add(c, 0);
      
    resolve_add(c, r);
  }
  else {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      r[index]=madlo(ua, a[index], 0);

    chain_t<> chain4;
    r[0]=chain4.madhi(ua, a_high, r[0]);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      r[index]=chain4.madhi(ua, a[index-1], r[index]);
    c=chain4.add(0, 0);

    chain_t<> chain5;
    chain5.sub(0, group_thread);
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++) {
      t=madlo(ub, b[index], 0);
      r[index]=chain5.sub(r[index], t);
    }
    c=chain5.add(c, 0);

    chain_t<> chain6;
    chain6.sub(0, group_thread);
    t=madhi(ub, b_high, 0);
    r[0]=chain6.sub(r[0], t);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++) {
      t=madhi(ub, b[index-1], 0);
      r[index]=chain6.sub(r[index], t);
    }
    c=chain6.add(c, 0);
  
    resolve_add(c, r);
    sign=__shfl_sync(sync, r[LIMBS-1], 31, TPI);
    if(sign<0) 
      fast_negate(r);
  }
}

template<class env>
__device__ __forceinline__ void core_t<env>::gcd(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
  uint32_t       sync=sync_mask();
  uint32_t       a_current[LIMBS], b_current[LIMBS], a_temp[LIMBS], b_temp[LIMBS];
  uint32_t       t, a_low, a_high, b_low, b_high;
  int32_t        delta, count, gcd_count;
  signed_coeff_t reducer;

  // NOTE: this algorithm is for unpadded values only
  
  gcd_count=ctz(a);
  if(gcd_count==BITS) {
    mpset<LIMBS>(r, b);
    return;
  }
  count=ctz(b);
  if(count==BITS) {
    mpset<LIMBS>(r, a);
    return;
  }
  shift_right(a_current, a, gcd_count);
  shift_right(b_current, b, count);
  gcd_count=umin(gcd_count, count);

  // done when top 26 bits of A and B are zero
  while(true) {
    a_high=__shfl_sync(sync, a_current[LIMBS-1], 31, TPI);
    b_high=__shfl_sync(sync, b_current[LIMBS-1], 31, TPI);
    if((a_high | b_high)<=0x3F)
      break;
    if(a_high>b_high)
      delta=1;
    else if(a_high<b_high)
      delta=-1;
    else {
      delta=compare(a_current, b_current);
      if(delta==0)
        break;
    }
    if(delta==1) {
      t=mpsub<LIMBS>(a_temp, a_current, b_current);
      fast_propagate_sub(t, a_temp);
      count=ctz(a_temp);
      rotate_right(a_current, a_temp, count);
    }
    else if(delta==-1) {
      t=mpsub<LIMBS>(b_temp, b_current, a_current);
      fast_propagate_sub(t, b_temp);
      count=ctz(b_temp);
      rotate_right(b_current, b_temp, count);
    }
  }

  // require that top 26 bits of A and B are 0
  while(true) {
    a_low=__shfl_sync(sync, a_current[0], 0, TPI);
    b_low=__shfl_sync(sync, b_current[0], 0, TPI);
    
    gcd_reduce(reducer, a_low, b_low);
    gcd_product(sync, a_temp, reducer.alpha_a, a_current, reducer.alpha_b, b_current);  
    gcd_product(sync, b_temp, reducer.beta_a, a_current, reducer.beta_b, b_current);

    a_low=__shfl_sync(sync, a_temp[0], 0, TPI);
    b_low=__shfl_sync(sync, b_temp[0], 0, TPI);

    if(a_low!=0) {
      t=__shfl_sync(sync, a_temp[0], threadIdx.x+1, TPI);
      mpright<LIMBS>(a_current, a_temp, uctz(a_low), t);
    }
    else {
      count=ctz(a_temp);
      if(count==BITS)
        break;
      rotate_right(a_current, a_temp, count);
    }
    
    if(b_low!=0) {
      t=__shfl_sync(sync, b_temp[0], threadIdx.x+1, TPI);
      mpright<LIMBS>(b_current, b_temp, uctz(b_low), t);
    }
    else {
      count=ctz(b_temp);
      if(count==BITS)
        break;
      rotate_right(b_current, b_temp, count);
    }
    
    if(reducer.beta_a==-1 && reducer.beta_b==1) 
      mpswap<LIMBS>(a_current, b_current);
  }

  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=a_temp[index] | b_temp[index];

  count=ctz(r);
  rotate_right(r, r, count-gcd_count+BITS);
}

} /* namespace cgbn */
