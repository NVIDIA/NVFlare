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
__device__ __forceinline__ void core_t<env>::sqrt_resolve_rem(uint32_t rem[LIMBS], const uint32_t s[LIMBS], const uint32_t top, const uint32_t r[LIMBS], const uint32_t shift) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t mask, phi[LIMBS], plo[LIMBS], t, s0[LIMBS];
  
  // remainder computation: r'=(2*s*(s mod b)+r)/b^2
  // where s=square root, r=remainder, b=2^shift.

  mask=(1<<(shift & 0x1F))-1;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) 
    if(index*32+32<=shift)
      s0[index]=__shfl_sync(sync, s[index], 0, TPI);
    else if(index*32>shift)
      s0[index]=0;
    else 
      s0[index]=__shfl_sync(sync, s[index], 0, TPI) & mask;
    
  mpadd<LIMBS>(s0, s0, s0);
  mpmul<LIMBS>(plo, phi, s, s0);
  
  chain_t<> chain;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    plo[index]=chain.add(plo[index], r[index]);
  t=(group_thread==TPI-1) ? top : 0;
  phi[0]=chain.add(phi[0], t);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    phi[index]=chain.add(phi[index], 0); 
    
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) 
    phi[index]=__shfl_sync(sync, phi[index], threadIdx.x-1, TPI);

  t=0;
  if(group_thread!=0) {
    t=mpadd<LIMBS>(plo, plo, phi);
    mpzero<LIMBS>(phi);
  }
  t=fast_propagate_add(t, plo);
  
  if(group_thread==0) 
    mpadd32<LIMBS>(phi, phi, t);
  bitwise_mask_select(plo, plo, phi, shift*2);
  rotate_right(rem, plo, shift*2);
}

template<class env>
__device__ __forceinline__ void core_t<env>::sqrt(uint32_t s[LIMBS], const uint32_t x[LIMBS], const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t dlo[DLIMBS], dhi[DLIMBS], rem[DLIMBS], dtemp[DLIMBS], approx[DLIMBS], t, c;
  uint32_t remainder[LIMBS], q[LIMBS], plo[LIMBS], phi[LIMBS];
  int32_t  top;
  
  dlimbs_scatter(dlo, x, TPI-2);
  dlimbs_scatter(dhi, x, TPI-1);
    
  top=dlimbs_sqrt_rem_wide(dtemp, rem, dlo, dhi);
  dlimbs_approximate(approx, dtemp);

  // set up remainder
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    t=__shfl_up_sync(sync, x[index], 1, TPI);
    remainder[index]=(group_thread==0) ? 0 : t;
  }
  dlimbs_gather(remainder, rem, TPI-1);
  
  // initialize s to be 2 * divisor, silent 1 at top of s
  t=__shfl_up_sync(sync, dtemp[DLIMBS-1], 1, TPI);
  t=(group_thread==0) ? 0 : t;
  mpleft<DLIMBS>(dtemp, dtemp, 1, t);
  mpzero<LIMBS>(s);
  dlimbs_gather(s, dtemp, TPI-1);

  // silent 1 at top of s
  #pragma nounroll
  for(int32_t index=TPI-2;index>=(int32_t)(TPI-numthreads);index--) {
    dlimbs_scatter(dtemp, remainder, TPI-1);
    dlimbs_sqrt_estimate(dtemp, top, dtemp, approx);
    dlimbs_all_gather(q, dtemp);
    
    if(group_thread==index)
      mpset<LIMBS>(s, q);

    // compute low/high
    mpmul<LIMBS>(plo, phi, s, q);

    // double q in s
    c=0;
    if(group_thread==index)
      c=mpadd<LIMBS>(s, s, q);
    c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
    s[0]=s[0]+c;

    c=mpsub<LIMBS>(remainder, remainder, phi);
    top=__shfl_sync(sync, remainder[0], TPI-1, TPI) - q[0];  // we subtract q[0] because of the silent 1 in s
    
    #pragma unroll
    for(int32_t limb=0;limb<LIMBS;limb++)                    // shuffle remainder up by 1
      remainder[limb]=__shfl_up_sync(sync, remainder[limb], 1, TPI);

    c=__shfl_up_sync(sync, c, 1, TPI);                       // shuffle carry up by 1
    c=(group_thread==0) ? 0 : c;
    c=c+mpsub<LIMBS>(remainder, remainder, plo);
    
    top=top+resolve_sub(c, remainder);

    while(top<0) {
      c=0;
      if(group_thread==index) {
        // decrement s by 2, if we borrow, need to resolve in next thread
        c=mpsub32<LIMBS>(s, s, 2);
      }
      c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
      s[0]=s[0]+c;
      
      add_cc(group_thread==index, 0xFFFFFFFF);
      #pragma unroll
      for(int32_t limb=0;limb<LIMBS;limb++)
        remainder[limb]=addc_cc(remainder[limb], s[limb]);
      c=addc(0, 0);
  
      top=top+1+fast_propagate_add(c, remainder);
    }
  }
  t=__shfl_down_sync(sync, s[0], 1, TPI);
  t=(group_thread==TPI-1) ? 1 : t;
  mpright<LIMBS>(s, s, 1, t);

  #pragma unroll
  for(int32_t limb=0;limb<LIMBS;limb++)
    s[limb]=__shfl_sync(sync, s[limb], threadIdx.x-numthreads, TPI);
}


template<class env>
__device__ __forceinline__ void core_t<env>::sqrt_rem(uint32_t s[LIMBS], uint32_t r[LIMBS], const uint32_t x[LIMBS], const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t dlo[DLIMBS], dhi[DLIMBS], rem[DLIMBS], dtemp[DLIMBS], approx[DLIMBS], t, c;
  uint32_t remainder[LIMBS], q[LIMBS], plo[LIMBS], phi[LIMBS];
  int32_t  top;
  
  dlimbs_scatter(dlo, x, TPI-2);
  dlimbs_scatter(dhi, x, TPI-1);
    
  top=dlimbs_sqrt_rem_wide(dtemp, rem, dlo, dhi);
  dlimbs_approximate(approx, dtemp);

  // set up remainder
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    t=__shfl_up_sync(sync, x[index], 1, TPI);
    remainder[index]=(group_thread==0) ? 0 : t;
  }
  dlimbs_gather(remainder, rem, TPI-1);
  
  // initialize s to be 2 * divisor, silent 1 at top of s
  t=__shfl_up_sync(sync, dtemp[DLIMBS-1], 1, TPI);
  t=(group_thread==0) ? 0 : t;
  mpleft<DLIMBS>(dtemp, dtemp, 1, t);
  mpzero<LIMBS>(s);
  dlimbs_gather(s, dtemp, TPI-1);

  // silent 1 at top of s
  #pragma nounroll
  for(int32_t index=TPI-2;index>=(int32_t)(TPI-numthreads);index--) {
    dlimbs_scatter(dtemp, remainder, TPI-1);
    dlimbs_sqrt_estimate(dtemp, top, dtemp, approx);
    dlimbs_all_gather(q, dtemp);
    
    if(group_thread==index)
      mpset<LIMBS>(s, q);

    // compute low/high
    mpmul<LIMBS>(plo, phi, s, q);

    // double q in s
    c=0;
    if(group_thread==index)
      c=mpadd<LIMBS>(s, s, q);
    c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
    s[0]=s[0]+c;

    c=mpsub<LIMBS>(remainder, remainder, phi);
    top=__shfl_sync(sync, remainder[0], TPI-1, TPI) - q[0];  // we subtract q[0] because of the silent 1 in s
    
    #pragma unroll
    for(int32_t limb=0;limb<LIMBS;limb++)                    // shuffle remainder up by 1
      remainder[limb]=__shfl_up_sync(sync, remainder[limb], 1, TPI);

    c=__shfl_up_sync(sync, c, 1, TPI);                       // shuffle carry up by 1
    c=(group_thread==0) ? 0 : c;
    c=c+mpsub<LIMBS>(remainder, remainder, plo);
    
    top=top+resolve_sub(c, remainder);

    while(top<0) {
      c=0;
      if(group_thread==index) {
        // decrement s by 2, if we borrow, need to resolve in next thread
        c=mpsub32<LIMBS>(s, s, 2);
      }
      c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
      s[0]=s[0]+c;
      
      add_cc(group_thread==index, 0xFFFFFFFF);
      #pragma unroll
      for(int32_t limb=0;limb<LIMBS;limb++)
        remainder[limb]=addc_cc(remainder[limb], s[limb]);
      c=addc(0, 0);
  
      top=top+1+fast_propagate_add(c, remainder);
    }
  }
  t=__shfl_down_sync(sync, s[0], 1, TPI);
  t=(group_thread==TPI-1) ? 1 : t;
  mpright<LIMBS>(s, s, 1, t);

  #pragma unroll
  for(int32_t limb=0;limb<LIMBS;limb++)
    s[limb]=__shfl_sync(sync, s[limb], threadIdx.x-numthreads, TPI);

  #pragma unroll
  for(int32_t limb=0;limb<LIMBS;limb++) 
    r[limb]=__shfl_sync(sync, remainder[limb], threadIdx.x-numthreads, TPI);
  
  if(group_thread>=numthreads) 
    mpzero<LIMBS>(r);
  if(group_thread==numthreads)
    r[0]=top;
}

template<class env>
__device__ __forceinline__ void core_t<env>::sqrt_wide(uint32_t s[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t dlo[DLIMBS], dhi[DLIMBS], rem[DLIMBS], dtemp[DLIMBS], approx[DLIMBS], t, c;
  uint32_t remainder[LIMBS], q[LIMBS], plo[LIMBS], phi[LIMBS];
  int32_t  top;
  
  dlimbs_scatter(dlo, hi, TPI-2);
  dlimbs_scatter(dhi, hi, TPI-1);
    
  top=dlimbs_sqrt_rem_wide(dtemp, rem, dlo, dhi);
  dlimbs_approximate(approx, dtemp);

  // set up remainder
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    remainder[index]=(group_thread==TPI-1) ? lo[index] : hi[index];
    remainder[index]=__shfl_sync(sync, remainder[index], threadIdx.x-1, TPI);
  }
  dlimbs_gather(remainder, rem, TPI-1);
    
  // initialize s to be 2 * divisor, silent 1 at top of s
  t=__shfl_up_sync(sync, dtemp[DLIMBS-1], 1, TPI);
  t=(group_thread==0) ? 0 : t;
  mpleft<DLIMBS>(dtemp, dtemp, 1, t);
  mpzero<LIMBS>(s);
  dlimbs_gather(s, dtemp, TPI-1);
  
  // silent 1 at top of s
  #pragma nounroll
  for(int32_t index=TPI-2;index>=(int32_t)(TPI-numthreads);index--) {
    dlimbs_scatter(dtemp, remainder, TPI-1);
    dlimbs_sqrt_estimate(dtemp, top, dtemp, approx);
    dlimbs_all_gather(q, dtemp);
    
    if(group_thread==index)
      mpset<LIMBS>(s, q);

    // compute low/high
    mpmul<LIMBS>(plo, phi, s, q);

    // double q in s
    c=0;
    if(group_thread==index)
      c=mpadd<LIMBS>(s, s, q);
    c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
    s[0]=s[0]+c;
    
    c=mpsub<LIMBS>(remainder, remainder, phi);
    top=__shfl_sync(sync, remainder[0], TPI-1, TPI) - q[0];  // we subtract q[0] because of the silent 1 in s

    #pragma unroll
    for(int32_t limb=0;limb<LIMBS;limb++) {                  // shuffle remainder up by 1
      t=__shfl_sync(sync, lo[limb], index, TPI);
      remainder[limb]=(group_thread==TPI-1) ? t : remainder[limb];
      remainder[limb]=__shfl_sync(sync, remainder[limb], threadIdx.x-1, TPI);
    }

    c=__shfl_up_sync(sync, c, 1, TPI);                       // shuffle carry up by 1
    c=(group_thread==0) ? 0 : c;
    c=c+mpsub<LIMBS>(remainder, remainder, plo);
    
    top=top+resolve_sub(c, remainder);

    while(top<0) {
      c=0;
      if(group_thread==index) {
        // decrement s by 2, if we borrow, need to resolve in next thread
        c=mpsub32<LIMBS>(s, s, 2);
      }
      c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
      s[0]=s[0]+c;
   
      add_cc(group_thread==index, 0xFFFFFFFF);
      #pragma unroll
      for(int32_t limb=0;limb<LIMBS;limb++)
        remainder[limb]=addc_cc(remainder[limb], s[limb]);
      c=addc(0, 0);
      
      top=top+1+fast_propagate_add(c, remainder);
    }
  }
  t=__shfl_down_sync(sync, s[0], 1, TPI);
  t=(group_thread==TPI-1) ? 1 : t;
  mpright<LIMBS>(s, s, 1, t);

  #pragma unroll
  for(int32_t limb=0;limb<LIMBS;limb++)
    s[limb]=__shfl_sync(sync, s[limb], threadIdx.x-numthreads, TPI);
}

template<class env>
__device__ __forceinline__ uint32_t core_t<env>::sqrt_rem_wide(uint32_t s[LIMBS], uint32_t r[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t dlo[DLIMBS], dhi[DLIMBS], rem[DLIMBS], dtemp[DLIMBS], approx[DLIMBS], t, c;
  uint32_t remainder[LIMBS], q[LIMBS], plo[LIMBS], phi[LIMBS];
  int32_t  top;
  
  dlimbs_scatter(dlo, hi, TPI-2);
  dlimbs_scatter(dhi, hi, TPI-1);
    
  top=dlimbs_sqrt_rem_wide(dtemp, rem, dlo, dhi);
  dlimbs_approximate(approx, dtemp);

  // set up remainder
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    remainder[index]=(group_thread==TPI-1) ? lo[index] : hi[index];
    remainder[index]=__shfl_sync(sync, remainder[index], threadIdx.x-1, TPI);
  }
  dlimbs_gather(remainder, rem, TPI-1);
    
  // initialize s to be 2 * divisor, silent 1 at top of s
  t=__shfl_up_sync(sync, dtemp[DLIMBS-1], 1, TPI);
  t=(group_thread==0) ? 0 : t;
  mpleft<DLIMBS>(dtemp, dtemp, 1, t);
  mpzero<LIMBS>(s);
  dlimbs_gather(s, dtemp, TPI-1);
  
  // silent 1 at top of s
  #pragma nounroll
  for(int32_t index=TPI-2;index>=(int32_t)(TPI-numthreads);index--) {
    dlimbs_scatter(dtemp, remainder, TPI-1);
    dlimbs_sqrt_estimate(dtemp, top, dtemp, approx);
    dlimbs_all_gather(q, dtemp);
    
    if(group_thread==index)
      mpset<LIMBS>(s, q);

    // compute low/high
    mpmul<LIMBS>(plo, phi, s, q);

    // double q in s
    c=0;
    if(group_thread==index)
      c=mpadd<LIMBS>(s, s, q);
    c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
    s[0]=s[0]+c;
    
    c=mpsub<LIMBS>(remainder, remainder, phi);
    top=__shfl_sync(sync, remainder[0], TPI-1, TPI) - q[0];  // we subtract q[0] because of the silent 1 in s

    #pragma unroll
    for(int32_t limb=0;limb<LIMBS;limb++) {                  // shuffle remainder up by 1
      t=__shfl_sync(sync, lo[limb], index, TPI);
      remainder[limb]=(group_thread==TPI-1) ? t : remainder[limb];
      remainder[limb]=__shfl_sync(sync, remainder[limb], threadIdx.x-1, TPI);
    }

    c=__shfl_up_sync(sync, c, 1, TPI);                       // shuffle carry up by 1
    c=(group_thread==0) ? 0 : c;
    c=c+mpsub<LIMBS>(remainder, remainder, plo);
    
    top=top+resolve_sub(c, remainder);

    while(top<0) {
      c=0;
      if(group_thread==index) {
        // decrement s by 2, if we borrow, need to resolve in next thread
        c=mpsub32<LIMBS>(s, s, 2);
      }
      c=__shfl_sync(sync, c, threadIdx.x-1, TPI);
      s[0]=s[0]+c;
   
      add_cc(group_thread==index, 0xFFFFFFFF);
      #pragma unroll
      for(int32_t limb=0;limb<LIMBS;limb++)
        remainder[limb]=addc_cc(remainder[limb], s[limb]);
      c=addc(0, 0);
      
      top=top+1+fast_propagate_add(c, remainder);
    }
  }
  t=__shfl_down_sync(sync, s[0], 1, TPI);
  t=(group_thread==TPI-1) ? 1 : t;
  mpright<LIMBS>(s, s, 1, t);

  #pragma unroll
  for(int32_t limb=0;limb<LIMBS;limb++)
    s[limb]=__shfl_sync(sync, s[limb], threadIdx.x-numthreads, TPI);


  #pragma unroll
  for(int32_t limb=0;limb<LIMBS;limb++) 
    r[limb]=__shfl_sync(sync, remainder[limb], threadIdx.x-numthreads, TPI);
  
  if(group_thread>=numthreads) 
    mpzero<LIMBS>(r);
  if(group_thread==numthreads)
    r[0]=top;
  return (numthreads==TPI) ? top : 0;
}

} /* namespace cgbn */
