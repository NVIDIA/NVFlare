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
  uint32_t t0, t1, t, carry0, carry1, ru[LIMBS], ra[LIMBS], rl[LIMBS];
  uint64_t sum;
  int32_t  threads=(PADDING!=0) ? (BITS/32+LIMBS-1)/LIMBS : TPI;
  
  if(PADDING!=0)
    mpzero<LIMBS>(rl);
  mpset<LIMBS>(ra, add);
  mpzero<LIMBS>(ru);
  
  carry0=0;
  carry1=0;
  #pragma nounroll
  for(int32_t row=0;row<threads;row++) {
    #pragma unroll
    for(int32_t l=0;l<LIMBS;l++) {
      t=__shfl_sync(sync, b[l], row, TPI);
      
      chain_t<> chain1;
      for(int index=0;index<LIMBS;index++) 
        ra[index]=chain1.xmadll(a[index], t, ra[index]);
      carry1=chain1.add(carry1, 0);
            
      ra[0]=add_cc(ra[0], carry0);
      carry0=addc(0, 0);

      chain_t<> chain2;
      for(int index=0;index<LIMBS;index++)
        ru[index]=chain2.xmadhl(a[index], t, ru[index]);
      
      chain_t<> chain3;
      t0=chain3.xmadlh(a[0], t, ru[0]);
      for(int32_t index=1;index<LIMBS;index++)
        ru[index-1]=chain3.xmadlh(a[index], t, ru[index]);
      ru[LIMBS-1]=chain3.add(0, 0);
      
      // on Maxwell and Pascal, you can't chain an ADD into an XMAD
      // so this isn't quite optimal
      t1=add_cc(t0<<16, ra[0]);
      carry0=addc(t0>>16, carry0);

      if(group_thread<threads-row)
        rl[l]=t1;
      rl[l]=__shfl_sync(sync, rl[l], threadIdx.x+1, TPI);
        
      chain_t<> chain4;
      for(int index=0;index<LIMBS-1;index++)
        ra[index]=chain4.xmadhh(a[index], t, ra[index+1]);
      ra[LIMBS-1]=chain4.xmadhh(a[LIMBS-1], t, 0);
      
      sum=((uint64_t)ra[LIMBS-1]) + ((uint64_t)carry1) + ((uint64_t)rl[l]);
      ra[LIMBS-1]=sum;
      carry1=sum>>32;
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
  uint32_t t0, t1, term, carry0, carry1, rl[LIMBS], ra[LIMBS], ru[LIMBS];
  uint64_t sum;
  int32_t  threads=(PADDING!=0) ? (BITS/32)/LIMBS : TPI;
      
  if(PADDING!=0)
    mpzero<LIMBS>(rl);

  mpset<LIMBS>(ra, add);
  mpzero<LIMBS>(ru);
  
  carry0=0;
  carry1=0;
  #pragma nounroll
  for(int32_t r=0;r<threads;r++) {
    #pragma unroll
    for(int32_t l=0;l<LIMBS;l++) {
      term=__shfl_sync(sync, b[l], r, TPI);

      chain_t<> chain1;
      for(int32_t index=0;index<LIMBS;index++) 
        ra[index]=chain1.xmadll(a[index], term, ra[index]);
      carry1=chain1.add(carry1, 0);

      ra[0]=add_cc(ra[0], carry0);
      carry0=addc(0, 0);
            
      chain_t<> chain2;
      for(int32_t index=0;index<LIMBS;index++)
        ru[index]=chain2.xmadhl(a[index], term, ru[index]);
    
      chain_t<> chain3;
      t0=chain3.xmadlh(a[0], term, ru[0]);
      for(int32_t index=1;index<LIMBS;index++)
        ru[index-1]=chain3.xmadlh(a[index], term, ru[index]);
      ru[LIMBS-1]=chain3.add(0, 0);

      // on Maxwell and Pascal, you can't chain an ADD into an XMAD
      // so this isn't quite optimal
      t1=add_cc(t0<<16, ra[0]);
      carry0=addc(t0>>16, carry0);
      
      t0=__shfl_sync(sync, t1, 0, TPI);
      if(group_thread==r)
        rl[l]=t0;
      t1=__shfl_down_sync(sync, t1, 1, TPI);
      t1=(group_thread==TPI-1) ? 0 : t1;
      
      chain_t<> chain4;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index++)
        ra[index]=chain4.xmadhh(a[index], term, ra[index+1]);
      ra[LIMBS-1]=chain4.xmadhh(a[LIMBS-1], term, 0);

      // pair of IADD3 instructions
      sum=((uint64_t)ra[LIMBS-1]) + ((uint64_t)t1) + ((uint64_t)carry1);
      ra[LIMBS-1]=sum;
      carry1=sum>>32;
    } 
  }
  
  if(BITS/32!=threads*LIMBS) {
    #pragma unroll
    for(int32_t l=0;l<BITS/32-threads*LIMBS;l++) {
      term=__shfl_sync(sync, b[l], threads, TPI);

      chain_t<> chain5;
      for(int32_t index=0;index<LIMBS;index++) 
        ra[index]=chain5.xmadll(a[index], term, ra[index]);
      carry1=chain5.add(carry1, 0);

      ra[0]=add_cc(ra[0], carry0);
      carry0=addc(0, 0);
            
      chain_t<> chain6;
      for(int32_t index=0;index<LIMBS;index++)
        ru[index]=chain6.xmadhl(a[index], term, ru[index]);
    
      chain_t<> chain7;
      t0=chain7.xmadlh(a[0], term, ru[0]);
      for(int32_t index=1;index<LIMBS;index++)
        ru[index-1]=chain7.xmadlh(a[index], term, ru[index]);
      ru[LIMBS-1]=chain7.add(0, 0);

      // on Maxwell and Pascal, you can't chain an ADD into an XMAD
      // so this isn't quite optimal
      t1=add_cc(t0<<16, ra[0]);
      carry0=addc(t0>>16, carry0);
      
      t0=__shfl_sync(sync, t1, 0, TPI);
      if(group_thread==threads)
        rl[l]=t0;
      t1=__shfl_down_sync(sync, t1, 1, TPI);
      t1=(group_thread==TPI-1) ? 0 : t1;
      
      chain_t<> chain8;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index++)
        ra[index]=chain8.xmadhh(a[index], term, ra[index+1]);
      ra[LIMBS-1]=chain8.xmadhh(a[LIMBS-1], term, 0);

      // pair of IADD3 instructions
      sum=((uint64_t)ra[LIMBS-1]) + ((uint64_t)t1) + ((uint64_t)carry1);
      ra[LIMBS-1]=sum;
      carry1=sum>>32;
    } 
  }
  
  #pragma unroll
  for(int32_t index=LIMBS-1;index>=1;index--) 
    ru[index]=uleft_wrap(ru[index-1], ru[index], 16);
  ru[0]=ru[0]<<16;

  mpadd32<LIMBS>(ru, ru, carry0);
    
  chain_t<> chain9;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) 
    ra[index]=chain9.add(ra[index], ru[index]);
  carry1=chain9.add(carry1, 0);

  fast_propagate_add(carry1, ra);
  
  mpset<LIMBS>(lo, rl);
  mpset<LIMBS>(hi, ra);
}

} /* namespace cgbn */