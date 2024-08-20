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
  uint32_t u1, ru[LIMBS], r1=0, ra[LIMBS], c=0, t, q;
  uint64_t sum;
  
  mpzero<LIMBS>(ra);
  mpzero<LIMBS>(ru);
  
  #pragma nounroll
  for(int32_t thread=0;thread<TPI;thread++) {
    #pragma unroll
    for(int32_t word=0;word<LIMBS;word++) {
      t=__shfl_sync(sync, b[word], thread, TPI);
      
      chain_t<LIMBS+1> chain1;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ru[index]=chain1.xmadlh(a[index], t, ru[index]);

      chain_t<LIMBS+1> chain2;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ru[index]=chain2.xmadhl(a[index], t, ru[index]);
      u1=chain2.add(0, 0);

      chain_t<LIMBS+1> chain3;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ra[index]=chain3.xmadll(a[index], t, ra[index]);
      r1=chain3.add(r1, 0);

      chain_t<LIMBS> chain4;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index++)
        ra[index+1]=chain4.xmadhh(a[index], t, ra[index+1]);
      r1=chain4.xmadhh(a[LIMBS-1], t, r1);

      // split u[0] and add it into r
      t=ru[0]<<16;
      ra[0]=add_cc(ra[0], t);
      t=ru[0]>>16;
      c=addc(c, t);

      q=__shfl_sync(sync, ra[0], 0, TPI)*np0; 

      // skip u[0]
      chain_t<LIMBS> chain5;
      #pragma unroll
      for(int32_t index=1;index<LIMBS;index++)
        ru[index]=chain5.xmadlh(n[index], q, ru[index]);
      u1=chain5.add(u1, 0);

      // skip u[0], shift
      chain_t<LIMBS> chain6;
      #pragma unroll
      for(int32_t index=1;index<LIMBS;index++)
        ru[index-1]=chain6.xmadhl(n[index], q, ru[index]);
      ru[LIMBS-1]=chain6.add(u1, 0);

      // push the carry along
      ra[1]=add_cc(ra[1], c);
      c=addc(0, 0);
      
      // handles four XMADs for the q * n0 terms
      ra[0]=madlo_cc(n[0], q, ra[0]);
      ra[1]=madhic_cc(n[0], q, ra[1]);
      c=addc(c, 0);
      
      t=__shfl_sync(sync, ra[0], threadIdx.x+1, TPI);
      chain_t<LIMBS> chain7;
      #pragma unroll
      for(int32_t index=1;index<LIMBS;index++)
        ra[index]=chain7.xmadll(n[index], q, ra[index]);  
      r1=chain7.add(r1, 0);
      
      ra[0]=ra[1];
      chain_t<LIMBS> chain8;  // should be limbs-1
      #pragma unroll
      for(int32_t index=1;index<LIMBS-1;index++)
        ra[index]=chain8.xmadhh(n[index], q, ra[index+1]);
      ra[LIMBS-1]=chain8.xmadhh(n[LIMBS-1], q, 0);

      sum=((uint64_t)ra[LIMBS-1]) + ((uint64_t)r1) + ((uint64_t)t);
      ra[LIMBS-1]=sum;
      r1=sum>>32;
    }
  }

  #pragma unroll
  for(int32_t index=LIMBS-1;index>0;index--)
    ru[index]=__byte_perm(ru[index-1], ru[index], 0x5432);
  ru[0]=__byte_perm(0, ru[0], 0x5432);

  chain_t<LIMBS+1> chain10;
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=chain10.add(ra[index], ru[index]);
  r1=chain10.add(r1, 0);

  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, r1, 1, TPI);
  
  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  chain_t<LIMBS+1> chain11;
  r[0]=chain11.add(r[0], t);
  r[1]=chain11.add(r[1], c);
  #pragma unroll
  for(int32_t index=2;index<LIMBS;index++)
    r[index]=chain11.add(r[index], 0);
  c=chain11.add(r1, 0);

  c=-fast_propagate_add(c, r);

  // compute -n
  t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

  chain_t<LIMBS+1> chain12;
  r[0]=chain12.add(r[0], ~t & c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain12.add(r[index], ~n[index] & c);
  c=chain12.add(0, 0);
  fast_propagate_add(c, r);
  clear_padding(r);
}

template<class env>
__device__ __forceinline__ void core_t<env>::mont_reduce_wide(uint32_t r[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t n[LIMBS], const uint32_t np0, const bool zero) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t t0, t1, t, q, carry0, carry1, ru[LIMBS], ra[LIMBS], top;
  uint64_t sum;
  
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    ra[index]=lo[index];
    ru[index]=0;
  }
  
  carry0=0;
  carry1=0;
  #pragma nounroll
  for(int32_t thread=0;thread<TPI;thread++) {
    #pragma unroll
    for(int32_t word=0;word<LIMBS;word++) {
      ra[0]=add_cc(ra[0], carry0);
      carry0=addc(0, 0);
      t=ra[0] + (ru[0]<<16);
      q=__shfl_sync(sync, t, 0, TPI)*np0;

      chain_t<LIMBS+1> chain1;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ra[index]=chain1.xmadll(n[index], q, ra[index]);
      carry1=chain1.add(carry1, 0);

      chain_t<LIMBS> chain2;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ru[index]=chain2.xmadhl(n[index], q, ru[index]);
      
      chain_t<LIMBS+1> chain3;
      t1=chain3.xmadlh(n[0], q, ru[0]);
      #pragma unroll
      for(int32_t index=1;index<LIMBS;index++)
        ru[index-1]=chain3.xmadlh(n[index], q, ru[index]);
      ru[LIMBS-1]=chain3.add(0, 0);
      
      chain_t<LIMBS> chain4;
      t0=ra[0];
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index++)
        ra[index]=chain4.xmadhh(n[index], q, ra[index+1]);
      ra[LIMBS-1]=chain4.xmadhh(n[LIMBS-1], q, 0);
      
      t=t1<<16;
      t0=add_cc(t0, t);
      t=t1>>16;
      carry0=addc(carry0, t);

      // shift right by 32 bits (top thread gets zero)
      t0=__shfl_sync(sync, t0, threadIdx.x+1, TPI);
      if(!zero) {
        top=__shfl_sync(sync, hi[word], thread, TPI);    
        t0=(group_thread==TPI-1) ? top : t0;
      }

      sum=((uint64_t)ra[LIMBS-1]) + ((uint64_t)t0) + ((uint64_t)carry1);
      ra[LIMBS-1]=sum;
      carry1=sum>>32;
    }
  }
  
  #pragma unroll
  for(int32_t index=LIMBS-1;index>=1;index--) 
    ru[index]=uleft_wrap(ru[index-1], ru[index], 16);
  ru[0]=ru[0]<<16;

  mpadd32<LIMBS>(ru, ru, carry0);
    
  chain_t<LIMBS+1> chain5;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) 
    ra[index]=chain5.add(ra[index], ru[index]);
  carry1=chain5.add(carry1, 0);

  carry1=fast_propagate_add(carry1, ra);
  
  if(!zero && carry1!=0) {
    t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

    chain_t<LIMBS+1> chain6;
    ra[0]=chain6.add(ra[0], ~t);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      ra[index]=chain6.add(ra[index], ~n[index]);
    carry1=chain6.add(0, 0);
    fast_propagate_add(carry1, ra);
    clear_padding(ra);
  }

  mpset<LIMBS>(r, ra);
}


/****************************************************************************************************]
 *  FIX FIX FIX - figure out why this code doesn't work
 *
 *  This code doesn't work, but it's more elegant than the other implementation. 
 *  Keep it around for now.  Note -- Q values are correct, wrap arounds are 0
 ****************************************************************************************************/
#if 0

template<class env>
__device__ __forceinline__ void core_t<env>::mont_mul(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t n[LIMBS], const uint32_t np0) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t u1, ru[LIMBS], r1=0, ra[LIMBS], c=0, t, q, t0, t1;
  uint64_t sum;
  
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    ra[index]=0;
    ru[index]=0;
  }
  
  #pragma nounroll
  for(int32_t thread=0;thread<TPI;thread++) {
    #pragma unroll
    for(int32_t word=0;word<LIMBS;word++) {
      t=__shfl_sync(sync, b[word], thread, TPI);
      
      chain_t<> chain1;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ru[index]=chain1.xmadlh(a[index], t, ru[index]);

      chain_t<> chain2;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ru[index]=chain2.xmadhl(a[index], t, ru[index]);
      u1=chain2.add(0, 0);

      chain_t<> chain3;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ra[index]=chain3.xmadll(a[index], t, ra[index]);
      r1=chain3.add(r1, 0);

      chain_t<> chain4;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index++)
        ra[index+1]=chain4.xmadhh(a[index], t, ra[index+1]);
      r1=chain4.xmadhh(a[LIMBS-1], t, r1);


      ra[0]=add_cc(ra[0], c);
      c=addc(0, 0);
      t=ra[0]+(ru[0]<<16);
      q=__shfl_sync(sync, t, 0, TPI)*np0;

//if(blockIdx.x==0 && threadIdx.x==0) printf("Q=%08X\n", q);

      chain_t<> chain01;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ra[index]=chain01.xmadll(n[index], q, ra[index]);
      r1=chain01.add(r1, 0);

      chain_t<> chain02;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        ru[index]=chain02.xmadhl(n[index], q, ru[index]);
      u1=chain02.add(u1, 0);

      chain_t<> chain03;
      t1=chain03.xmadlh(n[0], q, ru[0]);
      #pragma unroll
      for(int32_t index=1;index<LIMBS;index++)
        ru[index-1]=chain03.xmadlh(n[index], q, ru[index]);
      ru[LIMBS-1]=chain03.add(u1, 0);
      
      t=t1<<16;
      t0=add_cc(ra[0], t);
      t=t1>>16;
      c=addc(c, t);

      chain_t<> chain04;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index++)
        ra[index]=chain04.xmadhh(n[index], q, ra[index+1]);
      ra[LIMBS-1]=chain04.xmadhh(n[LIMBS-1], q, 0);
      
      // shift right by 32 bits (top thread gets zero)
      t=__shfl_sync(sync, t0, threadIdx.x+1, TPI);
      sum=((uint64_t)ra[LIMBS-1]) + ((uint64_t)r1) + ((uint64_t)t);
      ra[LIMBS-1]=sum;
      r1=sum>>32;
    }
  }
  
  #pragma unroll
  for(int32_t index=LIMBS-1;index>0;index--)
    ru[index]=__byte_perm(ru[index-1], ru[index], 0x5432);
  ru[0]=__byte_perm(0, ru[0], 0x5432);

  mpadd32<LIMBS>(ru, ru, c);
  
  chain_t<LIMBS+1> chain10;
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=chain10.add(ra[index], ru[index]);
  r1=chain10.add(r1, 0);

  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, r1, 1, TPI);
  
  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  chain_t<LIMBS+1> chain11;
  r[0]=chain11.add(r[0], t);
  r[1]=chain11.add(r[1], c);
  #pragma unroll
  for(int32_t index=2;index<LIMBS;index++)
    r[index]=chain11.add(r[index], 0);
  c=chain11.add(r1, 0);

  c=-fast_propagate_add(c, r);

  // compute -n
  t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

  chain_t<LIMBS+1> chain12;
  r[0]=chain12.add(r[0], ~t & c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain12.add(r[index], ~n[index] & c);
  c=chain12.add(0, 0);
  fast_propagate_add(c, r);
  clear_padding(r);
}
#endif

} /* namespace cgbn */