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
  uint32_t t0, t1, term0, term1, carry, rl[LIMBS], ra[LIMBS+2], ru[LIMBS+1];
  int32_t  threads=(PADDING!=0) ? (BITS/32+LIMBS-1)/LIMBS : TPI;

  if(PADDING!=0)
    mpzero<LIMBS>(rl);
  mpset<LIMBS>(ra, add);
  ra[LIMBS]=0;
  ra[LIMBS+1]=0;
  mpzero<LIMBS>(ru);
  ru[LIMBS]=0;
 
  carry=0;
  #pragma nounroll
  for(int32_t row=0;row<threads;row+=2) {
    #pragma unroll
    for(int32_t l=0;l<LIMBS*2;l+=2) {
      if(l<LIMBS) 
        term0=__shfl_sync(sync, b[l], row, TPI);
      else
        term0=__shfl_sync(sync, b[l-LIMBS], row+1, TPI);
      if(l+1<LIMBS)
        term1=__shfl_sync(sync, b[l+1], row, TPI);
      else
        term1=__shfl_sync(sync, b[l+1-LIMBS], row+1, TPI);

      chain_t<> chain1;                               // aligned:   T0 * A_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain1.madlo(a[index], term0, ra[index]);
        ra[index+1]=chain1.madhi(a[index], term0, ra[index+1]);
      }
      if(LIMBS%2==0)
        ra[LIMBS]=chain1.add(ra[LIMBS], 0);      
      
      chain_t<> chain2;                               // unaligned: T0 * A_odd
      t0=chain2.add(ra[0], carry);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain2.madlo(a[index+1], term0, ru[index]);
        ru[index+1]=chain2.madhi(a[index+1], term0, ru[index+1]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=chain2.add(0, 0);

      chain_t<> chain3;                               // unaligned: T1 * A_even
      t1=chain3.madlo(a[0], term1, ru[0]);
      carry=chain3.madhi(a[0], term1, ru[1]);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-2;index+=2) {
        ru[index]=chain3.madlo(a[index+2], term1, ru[index+2]);
        ru[index+1]=chain3.madhi(a[index+2], term1, ru[index+3]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=0;
      else 
        ru[LIMBS-2]=chain3.add(0, 0);
      ru[LIMBS-1+LIMBS%2]=0;

      chain_t<> chain4;                               // aligned:   T1 * A_odd
      t1=chain4.add(t1, ra[1]);
      #pragma unroll
      for(int32_t index=0;index<(int32_t)(LIMBS-3);index+=2) {
        ra[index]=chain4.madlo(a[index+1], term1, ra[index+2]);
        ra[index+1]=chain4.madhi(a[index+1], term1, ra[index+3]);
      }
      ra[LIMBS-2-LIMBS%2]=chain4.madlo(a[LIMBS-1-LIMBS%2], term1, ra[LIMBS-LIMBS%2]);
      ra[LIMBS-1-LIMBS%2]=chain4.madhi(a[LIMBS-1-LIMBS%2], term1, ra[LIMBS+1-LIMBS%2]);
      if(LIMBS%2==1)
        ra[LIMBS-1]=chain4.add(0, 0);

      if(l<LIMBS) {
        if(group_thread<threads-row)
          rl[l]=t0;
        rl[l]=__shfl_sync(sync, rl[l], threadIdx.x+1, TPI);
        t0=rl[l];
      }
      else {
        if(group_thread<threads-1-row)
          rl[l-LIMBS]=t0;
        rl[l-LIMBS]=__shfl_sync(sync, rl[l-LIMBS], threadIdx.x+1, TPI);
        t0=rl[l-LIMBS];
      }
      if(l+1<LIMBS) {
        if(group_thread<threads-row)
          rl[l+1]=t1;
        rl[l+1]=__shfl_sync(sync, rl[l+1], threadIdx.x+1, TPI);
        t1=rl[l+1];
      }
      else {
        if(group_thread<threads-1-row)
          rl[l+1-LIMBS]=t1;
        rl[l-LIMBS+1]=__shfl_sync(sync, rl[l+1-LIMBS], threadIdx.x+1, TPI);
        t1=rl[l+1-LIMBS];
      }
            
      ra[LIMBS-2]=add_cc(ra[LIMBS-2], t0);
      ra[LIMBS-1]=addc_cc(ra[LIMBS-1], t1);
      ra[LIMBS]=addc(0, 0);
    }
  }
  
  if(PADDING==0) 
    mpset<LIMBS>(r, rl);
  else {
    #pragma unroll
    for(int index=0;index<LIMBS;index++)
      r[index]=__shfl_sync(sync, rl[index], threadIdx.x-(threads+1 & 0xFFFE), TPI);
    clear_padding(r);
  }
}

template<class env>
__device__ __forceinline__ void core_t<env>::mul_wide(uint32_t lo[LIMBS], uint32_t hi[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t add[LIMBS]) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t t, t0, t1, term0, term1, carry, rl[LIMBS], ra[LIMBS+2], ru[LIMBS+1];
  int32_t  threads=(PADDING!=0) ? (BITS/32)/LIMBS & 0xFFFE : TPI;

  if(PADDING!=0)
    mpzero<LIMBS>(rl);
  mpset<LIMBS>(ra, add);
  ra[LIMBS]=0;
  ra[LIMBS+1]=0;
  mpzero<LIMBS>(ru);
  ru[LIMBS]=0;
    
  carry=0;
  #pragma nounroll
  for(int32_t r=0;r<threads;r+=2) {
    #pragma unroll
    for(int32_t l=0;l<LIMBS*2;l+=2) {
      if(l<LIMBS) 
        term0=__shfl_sync(sync, b[l], r, TPI);
      else
        term0=__shfl_sync(sync, b[l-LIMBS], r+1, TPI);
      if(l+1<LIMBS)
        term1=__shfl_sync(sync, b[l+1], r, TPI);
      else
        term1=__shfl_sync(sync, b[l+1-LIMBS], r+1, TPI);
        
      chain_t<> chain1;                               // aligned:   T0 * A_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain1.madlo(a[index], term0, ra[index]);
        ra[index+1]=chain1.madhi(a[index], term0, ra[index+1]);
      }
      if(LIMBS%2==0)
        ra[LIMBS]=chain1.add(ra[LIMBS], 0);      
      
      chain_t<> chain2;                               // unaligned: T0 * A_odd
      t0=chain2.add(ra[0], carry);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain2.madlo(a[index+1], term0, ru[index]);
        ru[index+1]=chain2.madhi(a[index+1], term0, ru[index+1]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=chain2.add(0, 0);
      
      chain_t<> chain3;                               // unaligned: T1 * A_even
      t1=chain3.madlo(a[0], term1, ru[0]);
      carry=chain3.madhi(a[0], term1, ru[1]);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-2;index+=2) {
        ru[index]=chain3.madlo(a[index+2], term1, ru[index+2]);
        ru[index+1]=chain3.madhi(a[index+2], term1, ru[index+3]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=0;
      else 
        ru[LIMBS-2]=chain3.add(0, 0);
      ru[LIMBS-1+LIMBS%2]=0;
      
      chain_t<> chain4;                               // aligned:   T1 * A_odd
      t1=chain4.add(t1, ra[1]);
      #pragma unroll
      for(int32_t index=0;index<(int32_t)(LIMBS-3);index+=2) {
        ra[index]=chain4.madlo(a[index+1], term1, ra[index+2]);
        ra[index+1]=chain4.madhi(a[index+1], term1, ra[index+3]);
      }
      ra[LIMBS-2-LIMBS%2]=chain4.madlo(a[LIMBS-1-LIMBS%2], term1, ra[LIMBS-LIMBS%2]);
      ra[LIMBS-1-LIMBS%2]=chain4.madhi(a[LIMBS-1-LIMBS%2], term1, ra[LIMBS+1-LIMBS%2]);
      if(LIMBS%2==1)
        ra[LIMBS-1]=chain4.add(0, 0);
          
      if(l<LIMBS) {
        t=__shfl_sync(sync, t0, 0, TPI);
        if(group_thread==r)
          rl[l]=t;
      }
      else {
        t=__shfl_sync(sync, t0, 0, TPI);
        if(group_thread==r+1)
          rl[l-LIMBS]=t;
      }
      if(l+1<LIMBS) {
        t=__shfl_sync(sync, t1, 0, TPI);
        if(group_thread==r)
          rl[l+1]=t;
      }
      else {
        t=__shfl_sync(sync, t1, 0, TPI);
        if(group_thread==r+1)
          rl[l-LIMBS+1]=t;
      }
      t0=__shfl_sync(sync, t0, threadIdx.x+1, TPI);
      t1=__shfl_sync(sync, t1, threadIdx.x+1, TPI);
      
      ra[LIMBS]=0;
      if(group_thread!=TPI-1) {
        ra[LIMBS-2]=add_cc(ra[LIMBS-2], t0);
        ra[LIMBS-1]=addc_cc(ra[LIMBS-1], t1);
        ra[LIMBS]=addc(0, 0);
      }
    }
  }

  chain_t<> chainXX;
  ra[0]=chainXX.add(ra[0], carry);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    ra[index]=chainXX.add(ra[index], ru[index-1]);
  carry=chainXX.add(ra[LIMBS], 0);

  /* use the imad algorithm to handle the tail */
  if(BITS/32>=threads*LIMBS+LIMBS) {
    #pragma unroll
    for(int32_t l=0;l<LIMBS;l++) {
      t=__shfl_sync(sync, b[l], threads, TPI);

      chain_t<> chain3;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ra[index]=chain3.madlo(a[index], t, ra[index]);
      carry=chain3.add(carry, 0);

      t0=__shfl_sync(sync, ra[0], 0, TPI);
      if(group_thread==threads) 
        rl[l]=t0;
        
      t1=__shfl_down_sync(sync, ra[0], 1, TPI);
      t1=(group_thread==TPI-1) ? 0 : t1;

      chain_t<> chain4;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index++)
        ra[index]=chain4.madhi(a[index], t, ra[index+1]);
      ra[LIMBS-1]=chain4.madhi(a[LIMBS-1], t, carry);
      carry=chain4.add(0, 0);
     
      ra[LIMBS-1]=add_cc(ra[LIMBS-1], t1);
      carry=addc(carry, 0);
    } 
  }

  if((BITS/32)%LIMBS!=0) {
    uint32_t r=threads+(BITS/32>=threads*LIMBS+LIMBS);
    
    #pragma unroll
    for(int32_t l=0;l<(BITS/32)%LIMBS;l++) {
      t=__shfl_sync(sync, b[l], r, TPI);

      chain_t<> chain3;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ra[index]=chain3.madlo(a[index], t, ra[index]);
      carry=chain3.add(carry, 0);

      t0=__shfl_sync(sync, ra[0], 0, TPI);
      if(group_thread==r) 
        rl[l]=t0;
        
      t1=__shfl_down_sync(sync, ra[0], 1, TPI);
      t1=(group_thread==TPI-1) ? 0 : t1;

      chain_t<> chain4;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index++)
        ra[index]=chain4.madhi(a[index], t, ra[index+1]);
      ra[LIMBS-1]=chain4.madhi(a[LIMBS-1], t, carry);
      carry=chain4.add(0, 0);
     
      ra[LIMBS-1]=add_cc(ra[LIMBS-1], t1);
      carry=addc(carry, 0);
    } 
  }

  mpset<LIMBS>(lo, rl);
  mpset<LIMBS>(hi, ra);
  
  fast_propagate_add(carry, hi);
}

} /* namespace cgbn */