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
__device__ __forceinline__ void core_t<env>::mul(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t add[LIMBS]) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t t, rh[LIMBS+1], rl[LIMBS];
  int32_t  threads=(PADDING!=0) ? (BITS/32+LIMBS-1)/LIMBS : TPI;

  if(PADDING!=0)
    mpzero<LIMBS>(rl);
  mpset<LIMBS>(rh, add);
  rh[LIMBS]=0;
    
  #pragma nounroll
  for(int32_t row=0;row<threads;row++) {
    #pragma unroll
    for(int32_t l=0;l<LIMBS;l++) {      
      t=__shfl_sync(sync, b[l], row, TPI);
      
      chain_t<> chain1;
      #pragma unroll
      for(int index=0;index<LIMBS;index++) 
        rh[index]=chain1.madlo(a[index], t, rh[index]);
      rh[LIMBS]=chain1.add(rh[LIMBS], 0);
      
      if(group_thread<threads-row)
        rl[l]=rh[0];
      rl[l]=__shfl_sync(sync, rl[l], threadIdx.x+1, TPI);

      chain_t<> chain2;
      #pragma unroll
      for(int index=0;index<LIMBS-1;index++)
        rh[index]=chain2.madhi(a[index], t, rh[index+1]);
      rh[LIMBS-1]=chain2.madhi(a[LIMBS-1], t, rh[LIMBS]);
      rh[LIMBS]=chain2.add(0, 0);
      
      rh[LIMBS-1]=add_cc(rh[LIMBS-1], rl[l]);
      rh[LIMBS]=addc(rh[LIMBS], 0);      
    }
  }
  
  if(PADDING==0) 
    mpset<LIMBS>(r, rl);
  else {
    #pragma unroll
    for(int index=0;index<LIMBS;index++)
      r[index]=__shfl_sync(sync, rl[index], threadIdx.x-threads, TPI);
    if(PAD_LIMB!=0 && group_thread==threads-1) {
      #pragma unroll
      for(int index=PAD_LIMB;index<LIMBS;index++)
        r[index]=0;
    }
  }
}

template<class env>
__device__ __forceinline__ void core_t<env>::mul_wide(uint32_t lo[LIMBS], uint32_t hi[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t add[LIMBS]) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t carry=0, low, t, rl[LIMBS], rh[LIMBS];
  int32_t  threads=(PADDING!=0) ? (BITS/32)/LIMBS : TPI;

  if(PADDING!=0)
    mpzero<LIMBS>(rl);

  mpset<LIMBS>(rh, add);
  
  #pragma nounroll
  for(int32_t r=0;r<threads;r++) {
    #pragma unroll
    for(int32_t l=0;l<LIMBS;l++) {
      t=__shfl_sync(sync, b[l], r, TPI);

      chain_t<> chain1;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        rh[index]=chain1.madlo(a[index], t, rh[index]);
      carry=chain1.add(carry, 0);

      low=__shfl_sync(sync, rh[0], 0, TPI);
      if(group_thread==r) 
        rl[l]=low;
        
      low=__shfl_down_sync(sync, rh[0], 1, TPI);
      low=(group_thread==TPI-1) ? 0 : low;
      
      chain_t<> chain2;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index++)
        rh[index]=chain2.madhi(a[index], t, rh[index+1]);
      rh[LIMBS-1]=chain2.madhi(a[LIMBS-1], t, carry);
      carry=chain2.add(0, 0);
     
      rh[LIMBS-1]=add_cc(rh[LIMBS-1], low);
      carry=addc(carry, 0);
    } 
  }

  if(BITS/32!=threads*LIMBS) {
    #pragma unroll
    for(int32_t l=0;l<BITS/32-threads*LIMBS;l++) {
      t=__shfl_sync(sync, b[l], threads, TPI);

      chain_t<> chain3;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        rh[index]=chain3.madlo(a[index], t, rh[index]);
      carry=chain3.add(carry, 0);

      low=__shfl_sync(sync, rh[0], 0, TPI);
      if(group_thread==threads) 
        rl[l]=low;
        
      low=__shfl_down_sync(sync, rh[0], 1, TPI);
      low=(group_thread==TPI-1) ? 0 : low;

      chain_t<> chain4;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index++)
        rh[index]=chain4.madhi(a[index], t, rh[index+1]);
      rh[LIMBS-1]=chain4.madhi(a[LIMBS-1], t, carry);
      carry=chain4.add(0, 0);
     
      rh[LIMBS-1]=add_cc(rh[LIMBS-1], low);
      carry=addc(carry, 0);
    } 
  }
  
  fast_propagate_add(carry, rh);
  
  mpset<LIMBS>(lo, rl);
  mpset<LIMBS>(hi, rh);
}

} /* namespace cgbn */