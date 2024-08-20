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
__device__ __forceinline__ void core_t<env>::mont_mul(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t n[LIMBS], const uint32_t np0) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t x[LIMBS], x1=0, x2, t, q, c;
  
  mpzero<LIMBS>(x);
  
  #pragma nounroll
  for(int32_t thread=0;thread<TPI;thread++) {
    #pragma unroll
    for(int word=0;word<LIMBS;word++) {
      t=__shfl_sync(sync, b[word], thread, TPI);
      
      chain_t<LIMBS+1> chain1;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        x[index]=chain1.madlo(a[index], t, x[index]);
      x1=chain1.add(x1, 0);
      
      chain_t<LIMBS+1> chain2;
      for(int32_t index=0;index<LIMBS-1;index++)
        x[index+1]=chain2.madhi(a[index], t, x[index+1]);
      x1=chain2.madhi(a[LIMBS-1], t, x1);
      x2=chain2.add(0, 0);
      
      q=__shfl_sync(sync, x[0], 0, TPI)*np0;

      chain_t<LIMBS+2> chain3;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        x[index]=chain3.madlo(n[index], q, x[index]);
      t=__shfl_sync(sync, x[0], threadIdx.x+1, TPI);
      x1=chain3.add(x1, t);
      x2=chain3.add(x2, 0);
      
      chain_t<LIMBS+1> chain4;
      for(int32_t index=0;index<LIMBS-1;index++)
        x[index]=chain4.madhi(n[index], q, x[index+1]);
      x[LIMBS-1]=chain4.madhi(n[LIMBS-1], q, x1);
      x1=chain4.add(x2, 0);
    }
  }
  
  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, x1, 1, TPI);
  
  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    x1=0;
  if(group_thread==0)
    t=0;

  chain_t<LIMBS+1> chain5;
  r[0]=chain5.add(x[0], t);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain5.add(x[index], 0);
  c=chain5.add(x1, 0);

  c=-fast_propagate_add(c, r);

  // compute -n
  t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

  chain_t<LIMBS+1> chain6;
  r[0]=chain6.add(r[0], ~t & c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain6.add(r[index], ~n[index] & c);
  c=chain6.add(0, 0);
  fast_propagate_add(c, r);
  clear_padding(r);
}

template<class env>
__device__ __forceinline__ void core_t<env>::mont_reduce_wide(uint32_t r[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t n[LIMBS], const uint32_t np0, const bool zero) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t c, x[LIMBS], t, q, top;
  
  mpset<LIMBS>(x, lo);
  
  c=0;
  #pragma nounroll
  for(int32_t thread=0;thread<TPI;thread++) {
    #pragma unroll
    for(int word=0;word<LIMBS;word++) {
      q=__shfl_sync(sync, x[0], 0, TPI)*np0;
      chain_t<LIMBS+1> chain1;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        x[index]=chain1.madlo(n[index], q, x[index]);
      c=chain1.add(c, 0);

      if(!zero)
        top=__shfl_sync(sync, hi[word], thread, TPI);
    
      // shift right by 32 bits (top thread gets zero)
      t=__shfl_sync(sync, x[0], threadIdx.x+1, TPI);
      if(!zero) {
        top=__shfl_sync(sync, hi[word], thread, TPI);    
        t=(group_thread==TPI-1) ? top : t;
      }

      chain_t<LIMBS+1> chain2;
      for(int32_t index=0;index<LIMBS-1;index++)
        x[index]=chain2.madhi(n[index], q, x[index+1]);
      x[LIMBS-1]=chain2.madhi(n[LIMBS-1], q, c);
      c=chain2.add(0, 0);
      
      x[LIMBS-1]=add_cc(x[LIMBS-1], t);
      c=addc(c, 0);
    }
  }
  
  c=fast_propagate_add(c, x);
  
  if(!zero && c!=0) {
    t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

    chain_t<LIMBS+1> chain3;
    x[0]=chain3.add(x[0], ~t);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      x[index]=chain3.add(x[index], ~n[index]);
    c=chain3.add(0, 0);
    fast_propagate_add(c, x);
  }
  mpset<LIMBS>(r, x);
}

} /* namespace cgbn */
