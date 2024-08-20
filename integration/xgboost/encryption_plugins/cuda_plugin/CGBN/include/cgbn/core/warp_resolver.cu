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

template<class core>
class dispatch_resolver_t<core, 32, 0> {
  public:
  static const uint32_t LIMBS=core::LIMBS;
  
  /****************************************************************
   * returns 0 if all bits are zero
   * returns -1 otherwise
   ****************************************************************/
  __device__ __forceinline__ static int32_t fast_negate(uint32_t &x) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t p, c;
  
    p=__ballot_sync(sync, x==0);
    c=(p+1^p)&lane;
    add_cc(c, 0xFFFFFFFF);
    x=subc(0, x);
    return (p==0xFFFFFFFF) ? 0 : 0xFFFFFFFF;
  }

  __device__ __forceinline__ static int32_t fast_negate(uint32_t x[LIMBS]) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t lor, p, c;
  
    lor=mplor<LIMBS>(x);
    p=__ballot_sync(sync, lor==0);
    c=(p+1^p)&lane;
    
    chain_t<> chain;
    chain.add(c, 0xFFFFFFFF);
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      x[index]=chain.sub(0, x[index]);
      
    return (p==0xFFFFFFFF) ? 0 : 0xFFFFFFFF;
  }

  /****************************************************************
   * returns 1 if carries out
   * returns 0 otherwise
   ****************************************************************/
  __device__ __forceinline__ static int32_t fast_propagate_add(const uint32_t carry, uint32_t &x) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c;
    uint64_t sum;
  
    g=__ballot_sync(sync, carry==1);
    p=__ballot_sync(sync, x==0xFFFFFFFF);
 
    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);
    
    x=x+(c!=0);
     
    return sum>>32;   // -(p==0xFFFFFFFF);
  }
  
  __device__ __forceinline__ static int32_t fast_propagate_add(const uint32_t carry, uint32_t x[LIMBS]) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t land, g, p, c;
    uint64_t sum;

    land=mpland<LIMBS>(x);
    g=__ballot_sync(sync, carry==1);
    p=__ballot_sync(sync, land==0xFFFFFFFF);

    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane & (p ^ sum);
    
    x[0]=add_cc(x[0], c!=0);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      x[index]=addc_cc(x[index], 0);
 
    return sum>>32; // -(p==0xFFFFFFFF);
  }


  /****************************************************************
   * returns 1 if borrows out
   * returns 0 otherwise
   ****************************************************************/

  __device__ __forceinline__ static int32_t fast_propagate_sub(const uint32_t carry, uint32_t &x) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c;
    uint64_t sum;
  
    g=__ballot_sync(sync, carry==0xFFFFFFFF);
    p=__ballot_sync(sync, x==0);

    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);

    x=x-(c!=0);
    return (sum>>32);     // -(p==0xFFFFFFFF);
  }

  __device__ __forceinline__ static int32_t fast_propagate_sub(const uint32_t carry, uint32_t x[LIMBS]) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c, lor;
    uint64_t sum;
  
    lor=mplor<LIMBS>(x);
    g=__ballot_sync(sync, carry==0xFFFFFFFF);
    p=__ballot_sync(sync, lor==0);

    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane & (p ^ sum);
    c=(c==0) ? 0 : 0xFFFFFFFF;
    
    x[0]=add_cc(x[0], c);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++) 
      x[index]=addc_cc(x[index], c);
    return sum>>32; // -(p==0xFFFFFFFF);
  }
  

  /****************************************************************
   * returns the high word 
   ****************************************************************/
  __device__ __forceinline__ static int32_t resolve_add(const int32_t carry, uint32_t &x) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c;
    uint64_t sum;
  
    c=__shfl_up_sync(sync, carry, 1);
    c=(warp_thread==0) ? 0 : c;
    x=add_cc(x, c);
    c=addc(0, 0);

    g=__ballot_sync(sync, c==1);
    p=__ballot_sync(sync, x==0xFFFFFFFF);

    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);
    x=x+(c!=0);
  
    c=carry+(sum>>32);
    return __shfl_sync(sync, c, 31);
  }

  __device__ __forceinline__ static int32_t resolve_add(const int32_t carry, uint32_t x[LIMBS]) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c, land;
    uint64_t sum;
    
    c=__shfl_up_sync(sync, carry, 1);
    c=(warp_thread==0) ? 0 : c;
    x[0]=add_cc(x[0], c);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++) 
      x[index]=addc_cc(x[index], 0);
    c=addc(0, 0);
  
    land=mpland<LIMBS>(x);
    g=__ballot_sync(sync, c==1);
    p=__ballot_sync(sync, land==0xFFFFFFFF);
  
    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);
    x[0]=add_cc(x[0], c!=0);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      x[index]=addc_cc(x[index], 0);
    
    c=carry+(sum>>32);
    return __shfl_sync(sync, c, 31);
  }

  /****************************************************************
   * returns the high word 
   ****************************************************************/
  __device__ __forceinline__ static int32_t resolve_sub(const int32_t carry, uint32_t &x) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p;
    int32_t  c;
    uint64_t sum;
  
    c=__shfl_up_sync(sync, carry, 1);
    c=(warp_thread==0) ? 0 : c;
    x=add_cc(x, c);
    c=addc(0, c>>31);

    g=__ballot_sync(sync, c==0xFFFFFFFF);
    p=__ballot_sync(sync, x==0);
  
    // wrap the carry around  
    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);

    x=x-(c!=0);
    c=carry-(sum>>32);
    return __shfl_sync(sync, c, 31);
  }
  
  __device__ __forceinline__ static int32_t resolve_sub(const int32_t carry, uint32_t x[LIMBS]) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, lor;
    int32_t  c;
    uint64_t sum;

    c=__shfl_up_sync(sync, carry, 1);
    c=(warp_thread==0) ? 0 : c;
    x[0]=add_cc(x[0], c);
    c=c>>31;
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++) 
      x[index]=addc_cc(x[index], c);
    c=addc(0, c);

    lor=mplor<LIMBS>(x);
    g=__ballot_sync(sync, c==0xFFFFFFFF);
    p=__ballot_sync(sync, lor==0);
  
    // wrap the carry around  
    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);
    c=(c==0) ? 0 : 0xFFFFFFFF;
    x[0]=add_cc(x[0], c);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++) 
      x[index]=addc_cc(x[index], c);

    c=carry-(sum>>32);
    return __shfl_sync(sync, c, 31);
  }  
};

} /* namespace cgbn */
