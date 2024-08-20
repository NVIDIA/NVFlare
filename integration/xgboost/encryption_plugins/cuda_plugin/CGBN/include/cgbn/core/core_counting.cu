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
__device__ __forceinline__ uint32_t core_t<env>::pop_count(const uint32_t a[LIMBS]) {
  uint32_t sync=sync_mask(), total=0;

  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    total+=__popc(a[index]);
  #pragma unroll
  for(int index=TPI/2;index>0;index=index>>1)
    total+=__shfl_xor_sync(sync, total, index, TPI);
  return total;
}

template<class env> 
__device__ __forceinline__ uint32_t core_t<env>::clz(const uint32_t a[LIMBS]) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1, warp_thread=threadIdx.x & warpSize-1;
  uint32_t clz, topclz;
  
  clz=mpclz<LIMBS>(a);
  topclz=__ballot_sync(sync, clz!=32*LIMBS);
  if(TPI<warpSize)
    topclz=topclz<<(warpSize-TPI)-(warp_thread-group_thread);
  topclz=uclz(topclz);
  if(topclz>=TPI)
    return BITS;
  return __shfl_sync(sync, (TPI-1-group_thread)*32*LIMBS + clz, 31-topclz, TPI)-LIMBS*TPI*32+BITS;
}

template<class env> 
__device__ __forceinline__ uint32_t core_t<env>::ctz(const uint32_t a[LIMBS]) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1, warp_thread=threadIdx.x & warpSize-1;
  uint32_t ctz, bottomctz;

  ctz=mpctz<LIMBS>(a);
  bottomctz=__ballot_sync(sync, ctz!=32*LIMBS);
  if(TPI<warpSize)
    bottomctz=bottomctz>>(warp_thread^group_thread);
  bottomctz=uctz(bottomctz);
  if(bottomctz>=TPI)
    return BITS;
  return __shfl_sync(sync, group_thread*32*LIMBS + ctz, bottomctz, TPI);
}

template<class env> 
__device__ __forceinline__ uint32_t core_t<env>::clzt(const uint32_t a[LIMBS]) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1, warp_thread=threadIdx.x & warpSize-1;
  uint32_t lor, topclz;

  lor=mplor<LIMBS>(a);
  topclz=__ballot_sync(sync, lor!=0);
  if(TPI<warpSize)
    topclz=topclz<<(warpSize-TPI)-(warp_thread-group_thread);
  topclz=uclz(topclz);
  return umin(topclz, TPI);
}

template<class env> 
__device__ __forceinline__ uint32_t core_t<env>::ctzt(const uint32_t a[LIMBS]) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1, warp_thread=threadIdx.x & warpSize-1;
  uint32_t lor, bottomctz;

  lor=mplor<LIMBS>(a);
  bottomctz=__ballot_sync(sync, lor!=0);
  if(TPI<warpSize)
    bottomctz=bottomctz>>(warp_thread^group_thread);
  bottomctz=uctz(bottomctz);
  return umin(topctz, TPI);
}

} /* namespace cgbn */