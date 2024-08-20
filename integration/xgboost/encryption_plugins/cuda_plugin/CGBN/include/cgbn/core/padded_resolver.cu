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

template<class core, uint32_t tpi, uint32_t padding>
class dispatch_resolver_t {
  public:
  static const uint32_t BITS=core::BITS;
  static const uint32_t LIMBS=core::LIMBS;
  
  static const uint32_t PAD_THREAD=core::PAD_THREAD;
  static const uint32_t PAD_LIMB=core::PAD_LIMB;
  
  static const uint32_t tpi_ONES=(1ull<<tpi)-1;
  static const uint32_t GROUP_ONES=0xFFFFFFFF / tpi_ONES;

  /****************************************************************
   * returns 1 if all bits are zero
   * returns 0 otherwise
   ****************************************************************/
  __device__ __forceinline__ static int32_t fast_negate(uint32_t &x) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & tpi-1, warp_thread=threadIdx.x & warpSize-1;
    uint32_t lane=1<<warp_thread;
    uint32_t p, c;
    int32_t  result;
    
    p=__ballot_sync(sync, x==0);
    c=(p+GROUP_ONES^p)&lane;
    add_cc(c, 0xFFFFFFFF);
    x=subc_cc(0, x);
    result=__shfl_sync(sync, x, padding, tpi);
    x=(group_thread<padding) ? x : 0;
    return result;
  }
  
  __device__ __forceinline__ static int32_t fast_negate(uint32_t x[LIMBS]) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & tpi-1, group_base=group_thread*LIMBS;
    uint32_t warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t lor, p, c;
    int32_t  result;
    
    lor=mplor<LIMBS>(x);
    p=__ballot_sync(sync, lor==0);
    c=(p+GROUP_ONES^p)&lane;
    
    chain_t<> chain;
    chain.add(c, 0xFFFFFFFF);
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      x[index]=chain.sub(0, x[index]);
   
    result=__shfl_sync(sync, x[PAD_LIMB], PAD_THREAD, tpi);
    
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      x[index]=(group_base<BITS/32-index) ? x[index] : 0;
      
    return result;
  }
        
  /****************************************************************
   * returns 1 if carries out
   * returns 0 otherwise
   ****************************************************************/
  __device__ __forceinline__ static int32_t fast_propagate_add(const uint32_t carry, uint32_t &x) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & tpi-1, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c;
    int32_t  result;
  
    g=__ballot_sync(sync, carry==1);
    p=__ballot_sync(sync, x==0xFFFFFFFF);
  
    c=lane & (g+g+p ^ p);
    x=x+(c!=0);
  
    result=__shfl_sync(sync, x, padding, tpi);
    x=(group_thread<padding) ? x : 0;
    return result;
  }
  
  __device__ __forceinline__ static int32_t fast_propagate_add(const uint32_t carry, uint32_t x[LIMBS]) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & tpi-1, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t land, g, p, c;
    int32_t  result;
    
    land=mpland<LIMBS>(x);
    g=__ballot_sync(sync, carry==1);
    p=__ballot_sync(sync, land==0xFFFFFFFF);
  
    c=lane & (g+g+p ^ p);
    
    x[0]=add_cc(x[0], c!=0);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      x[index]=addc_cc(x[index], 0);
  
    result=__shfl_sync(sync, x[PAD_LIMB], PAD_THREAD, tpi);
    x[PAD_LIMB]=(group_thread!=PAD_THREAD) ? x[PAD_LIMB] : 0;
    return result;
  }
  
  /****************************************************************
   * returns 1 if carries out
   * returns 0 otherwise
   ****************************************************************/
  __device__ __forceinline__ static int32_t fast_propagate_sub(const uint32_t carry, uint32_t &x) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & tpi-1;
    uint32_t lane=1<<(threadIdx.x & warpSize-1);
    uint32_t g, p, c;
    int32_t  result;
    
    g=__ballot_sync(sync, carry==0xFFFFFFFF);
    p=__ballot_sync(sync, x==0);
  
    c=lane & (g+g+p ^ p);
    x=x-(c!=0);
    result=__shfl_sync(sync, x, padding, tpi);
    x=(group_thread<padding) ? x : 0;
    return result & 1;
  }
  
  __device__ __forceinline__ static int32_t fast_propagate_sub(const uint32_t carry, uint32_t x[LIMBS]) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & tpi-1, group_base=group_thread*LIMBS;
    uint32_t warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t lor, g, p, c;
    int32_t  result;
     
    lor=mplor<LIMBS>(x);
    g=__ballot_sync(sync, carry==0xFFFFFFFF);
    p=__ballot_sync(sync, lor==0);
  
    c=lane & (g+g+p ^ p);
    c=(c==0) ? 0 : 0xFFFFFFFF;
 
    x[0]=add_cc(x[0], c);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      x[index]=addc_cc(x[index], c);
  
    result=__shfl_sync(sync, x[PAD_LIMB], PAD_THREAD, tpi);
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++) 
      x[index]=(group_base<BITS/32-index) ? x[index] : 0;
    return result & 1;
  }

  /****************************************************************
   * returns the high word 
   ****************************************************************/
  __device__ __forceinline__ static int32_t resolve_add(const int32_t carry, uint32_t &x) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & tpi-1, lane=1<<(threadIdx.x & warpSize-1);
    uint32_t g, p, c;
    int32_t  result;
  
    c=__shfl_sync(sync, carry, threadIdx.x-1, tpi);
    c=(group_thread==0) ? 0 : c;
    x=add_cc(x, c);
    c=addc(0, 0);

    g=__ballot_sync(sync, c==1);
    p=__ballot_sync(sync, x==0xFFFFFFFF);
  
    c=lane & (g+g+p ^ p);
    x=x+(c!=0);

    result=__shfl_sync(sync, x, padding, tpi);
    x=(group_thread<padding) ? x : 0;
    return result;
  }

  __device__ __forceinline__ static int32_t resolve_add(const int32_t carry, uint32_t x[LIMBS]) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & tpi-1, lane=1<<(threadIdx.x & warpSize-1);
    uint32_t g, p, c, land;
    int32_t  result;
    
    c=__shfl_sync(sync, carry, threadIdx.x-1, tpi);
    c=(group_thread==0) ? 0 : c;
    x[0]=add_cc(x[0], c);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++) 
      x[index]=addc_cc(x[index], 0);
    c=addc(0, 0);
  
    land=mpland<LIMBS>(x);
    g=__ballot_sync(sync, c==1);
    p=__ballot_sync(sync, land==0xFFFFFFFF);
  
    c=lane & (g+g+p ^ p);
    
    x[0]=add_cc(x[0], c!=0);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      x[index]=addc_cc(x[index], 0);
    
    result=__shfl_sync(sync, x[PAD_LIMB], PAD_THREAD, tpi);
    x[PAD_LIMB]=(group_thread!=PAD_THREAD) ? x[PAD_LIMB] : 0;
    return result;
  }
  
  /****************************************************************
   * returns the high word 
   ****************************************************************/
  __device__ __forceinline__ static int32_t resolve_sub(const int32_t carry, uint32_t &x) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & tpi-1, lane=1<<(threadIdx.x & warpSize-1);
    uint32_t g, p;
    int32_t  c;
    int32_t  result;
  
    c=__shfl_sync(sync, carry, threadIdx.x-1, tpi);
    c=(group_thread==0) ? 0 : c;
    x=add_cc(x, c);
    c=addc(0, c>>31);

    g=__ballot_sync(sync, c==0xFFFFFFFF);
    p=__ballot_sync(sync, x==0);
  
    c=lane & (g+g+p ^ p);
    x=x-(c!=0);

    result=__shfl_sync(sync, x, padding, tpi);
    x=(group_thread<padding) ? x : 0;
    return result;
  }

  __device__ __forceinline__ static int32_t resolve_sub(const int32_t carry, uint32_t x[LIMBS]) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & tpi-1, group_base=group_thread*LIMBS;
    uint32_t warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, land;
    int32_t  c;
    int32_t  result;
    
    if(group_thread==PAD_THREAD)
      x[PAD_LIMB]=x[PAD_LIMB]+2;

    c=__shfl_sync(sync, carry, threadIdx.x-1, tpi);
    c=(group_thread==0) ? 0 : c;
    x[0]=add_cc(x[0], c);
    c=c>>31;
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++) 
      x[index]=addc_cc(x[index], c);
    c=addc(0, c);
  
    lor=mplor<limbs>(x);
    g=__ballot_sync(sync, c==0xFFFFFFFF);
    p=__ballot_sync(sync, lor==0);
  
    c=lane & (g+g+p ^ p);
  
    c=(c==0) ? 0 : 0xFFFFFFFF;
    x[0]=add_cc(x[0], c);
    #pragma unroll
    for(int32_t index=1;index<limbs;index++) 
      x[index]=addc_cc(x[index], c);
    
    result=__shfl_sync(sync, x[PAD_LIMB], PAD_THREAD, tpi);
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++) 
      x[index]=(group_base<BITS/32-index) ? x[index] : 0;
    return result;
  }
  
};

} /* namespace cgbn */

