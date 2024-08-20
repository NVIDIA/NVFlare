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
__device__ __forceinline__ void core_t<env>::bitwise_and(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=a[index] & b[index];
}

template<class env> 
__device__ __forceinline__ void core_t<env>::bitwise_ior(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=a[index] | b[index];
}

template<class env> 
__device__ __forceinline__ void core_t<env>::bitwise_xor(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=a[index] ^ b[index];
}

template<class env> 
__device__ __forceinline__ void core_t<env>::bitwise_complement(uint32_t r[LIMBS], const uint32_t a[LIMBS]) {
  if(PADDING==0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      r[index]=~a[index];
  }
  else {
    uint32_t group_thread=threadIdx.x & TPI-1;
    
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      r[index]=(group_thread*LIMBS<BITS/32-index) ? ~a[index] : 0;
  }
}

template<class env> 
__device__ __forceinline__ void core_t<env>::bitwise_select(uint32_t r[LIMBS], const uint32_t clear[LIMBS], const uint32_t set[LIMBS], const uint32_t select[LIMBS]) {
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=(set[index] & select[index]) | (clear[index] & ~select[index]);
}

} /* namespace cgbn */

