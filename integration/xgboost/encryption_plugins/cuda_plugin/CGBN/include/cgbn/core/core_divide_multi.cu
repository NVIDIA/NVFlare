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
__device__ __forceinline__ void core_t<env>::div_wide(uint32_t q[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t denom[LIMBS], const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  int32_t  x2;
  uint32_t dtemp[DLIMBS], approx[DLIMBS], estimate[DLIMBS], t, c, x0, x1, d0, d1, correction;
  uint32_t x[LIMBS], y[LIMBS], plo[LIMBS], phi[LIMBS], quotient[LIMBS];
  
  mpzero<LIMBS>(quotient);
  if(numthreads<TPI) {
    chain_t<> chain;
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++) {
      x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
      x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI);
      y[index]=chain.sub(x[index], denom[index]);
    }   
    c=chain.sub(0, 0);
    
    if(resolve_sub(c, y)==0) {
      mpset<LIMBS>(x, y);
      quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
    }
  }
  else
    mpset<LIMBS>(x, hi);
    
  d1=__shfl_sync(sync, denom[LIMBS-1], TPI-1, TPI);
  d0=__shfl_sync(sync, denom[LIMBS-2], TPI-1, TPI);

  dlimbs_scatter(dtemp, denom, TPI-1);  
  dlimbs_approximate(approx, dtemp);
    
  // main loop that discovers the quotient
  #pragma nounroll
  for(int32_t thread=numthreads-1;thread>=0;thread--) {
    dlimbs_scatter(dtemp, x, TPI-1);
    dlimbs_div_estimate(estimate, dtemp, approx);
    dlimbs_all_gather(y, estimate);
      
    mpmul<LIMBS>(plo, phi, y, denom);
    c=mpsub<LIMBS>(x, x, phi);
    x2=__shfl_sync(sync, x[0], TPI-1, TPI);

    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++) {       // shuffle x up by 1
      t=__shfl_sync(sync, lo[index], thread, TPI);
      x[index]=__shfl_up_sync(sync, x[index], 1, TPI);
      x[index]=(group_thread==0) ? t : x[index];
    }
  
    c=__shfl_up_sync(sync, c, 1, TPI);                // shuffle carry up by 1
    c=(group_thread==0) ? 0 : c;
    c=c+mpsub<LIMBS>(x, x, plo);
    
    x2=x2+resolve_sub(c, x);
    x1=__shfl_sync(sync, x[LIMBS-1], TPI-1, TPI);
    x0=__shfl_sync(sync, x[LIMBS-2], TPI-1, TPI);
      
    correction=ucorrect(x0, x1, x2, d0, d1);
    if(correction!=0) {
      c=mpmul32<LIMBS>(plo, denom, correction);
      t=resolve_add(c, plo);
      c=mpadd<LIMBS>(x, x, plo);
      x2=x2+t+fast_propagate_add(c, x);
    }
    if(x2<0) {
      // usually the case
      c=mpadd<LIMBS>(x, x, denom);
      fast_propagate_add(c, x);
      correction++;
    }
    if(group_thread==thread)
      mpsub32<LIMBS>(quotient, y, correction);
  }
  mpset<LIMBS>(q, quotient);
}

template<class env> 
__device__ __forceinline__ void core_t<env>::rem_wide(uint32_t r[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t denom[LIMBS], const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  int32_t  x2;
  uint32_t dtemp[DLIMBS], approx[DLIMBS], estimate[DLIMBS], t, c, x0, x1, d0, d1, correction;
  uint32_t x[LIMBS], y[LIMBS], plo[LIMBS], phi[LIMBS];
  
  if(numthreads<TPI) {
    chain_t<> chain;
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++) {
      x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
      x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI);
      y[index]=chain.sub(x[index], denom[index]);
    }
    c=chain.sub(0, 0);
    if(resolve_sub(c, y)==0)
      mpset<LIMBS>(x, y);
  }
  else
    mpset<LIMBS>(x, hi);
    
  d1=__shfl_sync(sync, denom[LIMBS-1], TPI-1, TPI);
  d0=__shfl_sync(sync, denom[LIMBS-2], TPI-1, TPI);

  dlimbs_scatter(dtemp, denom, TPI-1);  
  dlimbs_approximate(approx, dtemp);
    
  // main loop that discovers the quotient
  #pragma nounroll
  for(int32_t thread=numthreads-1;thread>=0;thread--) {
    dlimbs_scatter(dtemp, x, TPI-1);
    dlimbs_div_estimate(estimate, dtemp, approx);
    dlimbs_all_gather(y, estimate);
      
    mpmul<LIMBS>(plo, phi, y, denom);
    c=mpsub<LIMBS>(x, x, phi);
    x2=__shfl_sync(sync, x[0], TPI-1, TPI);

    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++) {       // shuffle x up by 1
      t=__shfl_sync(sync, lo[index], thread, TPI);
      x[index]=__shfl_up_sync(sync, x[index], 1, TPI);
      x[index]=(group_thread==0) ? t : x[index];
    }
  
    c=__shfl_up_sync(sync, c, 1, TPI);                // shuffle carry up by 1
    c=(group_thread==0) ? 0 : c;
    c=c+mpsub<LIMBS>(x, x, plo);
    
    x2=x2+resolve_sub(c, x);
    x1=__shfl_sync(sync, x[LIMBS-1], TPI-1, TPI);
    x0=__shfl_sync(sync, x[LIMBS-2], TPI-1, TPI);
      
    correction=ucorrect(x0, x1, x2, d0, d1);
    if(correction!=0) {
      c=mpmul32<LIMBS>(plo, denom, correction);
      t=resolve_add(c, plo);
      c=mpadd<LIMBS>(x, x, plo);
      x2=x2+t+fast_propagate_add(c, x);
    }
    if(x2<0) {
      // usually the case
      c=mpadd<LIMBS>(x, x, denom);
      fast_propagate_add(c, x);
    }
  }
  mpset<LIMBS>(r, x);
}

template<class env>
__device__ __forceinline__ void core_t<env>::div_rem_wide(uint32_t q[LIMBS], uint32_t r[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t denom[LIMBS], const uint32_t numthreads) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  int32_t  x2;
  uint32_t dtemp[DLIMBS], approx[DLIMBS], estimate[DLIMBS], t, c, x0, x1, d0, d1, correction;
  uint32_t x[LIMBS], y[LIMBS], plo[LIMBS], phi[LIMBS], quotient[LIMBS];
  
  mpzero<LIMBS>(quotient);
  if(numthreads<TPI) {
    chain_t<> chain;
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++) {
      x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
      x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI);
      y[index]=chain.sub(x[index], denom[index]);
    }   
    c=chain.sub(0, 0);
    
    if(resolve_sub(c, y)==0) {
      mpset<LIMBS>(x, y);
      quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
    }
  }
  else
    mpset<LIMBS>(x, hi);
    
  d1=__shfl_sync(sync, denom[LIMBS-1], TPI-1, TPI);
  d0=__shfl_sync(sync, denom[LIMBS-2], TPI-1, TPI);

  dlimbs_scatter(dtemp, denom, TPI-1);  
  dlimbs_approximate(approx, dtemp);
    
  // main loop that discovers the quotient
  #pragma nounroll
  for(int32_t thread=numthreads-1;thread>=0;thread--) {
    dlimbs_scatter(dtemp, x, TPI-1);
    dlimbs_div_estimate(estimate, dtemp, approx);
    dlimbs_all_gather(y, estimate);
      
    mpmul<LIMBS>(plo, phi, y, denom);
    c=mpsub<LIMBS>(x, x, phi);
    x2=__shfl_sync(sync, x[0], TPI-1, TPI);

    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++) {       // shuffle x up by 1
      t=__shfl_sync(sync, lo[index], thread, TPI);
      x[index]=__shfl_up_sync(sync, x[index], 1, TPI);
      x[index]=(group_thread==0) ? t : x[index];
    }
  
    c=__shfl_up_sync(sync, c, 1, TPI);                // shuffle carry up by 1
    c=(group_thread==0) ? 0 : c;
    c=c+mpsub<LIMBS>(x, x, plo);
    
    x2=x2+resolve_sub(c, x);
    x1=__shfl_sync(sync, x[LIMBS-1], TPI-1, TPI);
    x0=__shfl_sync(sync, x[LIMBS-2], TPI-1, TPI);
      
    correction=ucorrect(x0, x1, x2, d0, d1);
    if(correction!=0) {
      c=mpmul32<LIMBS>(plo, denom, correction);
      t=resolve_add(c, plo);
      c=mpadd<LIMBS>(x, x, plo);
      x2=x2+t+fast_propagate_add(c, x);
    }
    if(x2<0) {
      // usually the case
      c=mpadd<LIMBS>(x, x, denom);
      fast_propagate_add(c, x);
      correction++;
    }
    if(group_thread==thread)
      mpsub32<LIMBS>(quotient, y, correction);
  }
  mpset<LIMBS>(q, quotient);
  mpset<LIMBS>(r, x);
}

} /* namespace cgbn */
