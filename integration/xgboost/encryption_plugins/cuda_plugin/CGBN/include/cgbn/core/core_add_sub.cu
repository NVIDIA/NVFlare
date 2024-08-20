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
__device__ __forceinline__ int32_t core_t<env>::add(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
  uint32_t carry;

  chain_t<> chain;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=chain.add(a[index], b[index]);
  carry=chain.add(0, 0);
  return fast_propagate_add(carry, r);
}

template<class env> 
__device__ __forceinline__ int32_t core_t<env>::sub(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
  uint32_t carry;
  
  chain_t<> chain;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=chain.sub(a[index], b[index]);
  carry=chain.sub(0, 0);
  return -fast_propagate_sub(carry, r);
} 

template<class env> 
__device__ __forceinline__ int32_t core_t<env>::negate(uint32_t r[LIMBS], const uint32_t a[LIMBS]) {
  mpset<LIMBS>(r, a);
  return fast_negate(r);
} 

} /* namespace cgbn */
