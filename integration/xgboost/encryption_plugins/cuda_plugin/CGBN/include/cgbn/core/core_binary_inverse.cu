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
__device__ __forceinline__ void core_t<env>::binary_inverse(uint32_t r[LIMBS], const uint32_t x[LIMBS]) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t inverse[LIMBS], c1, current[LIMBS], inv, t, low;
  uint32_t stop_thread;
  
  inv=-ubinary_inverse(__shfl_sync(sync, x[0], 0, TPI));

  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    current[index]=0xFFFFFFFF;

  if(PADDING==0) 
    stop_thread=TPI;
  else
    stop_thread=((BITS/32)+LIMBS-1)/LIMBS;
    
  c1=0;
  #pragma nounroll
  for(int32_t thread=0;thread<=stop_thread;thread++) {
    #pragma unroll
    for(int32_t limb=0;limb<LIMBS;limb++) {
      t=inv*__shfl_sync(sync, current[0], 0, TPI);

      if(group_thread==thread)
        inverse[limb]=t;

      chain_t<> chain1;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        current[index]=chain1.madlo(t, x[index], current[index]);
      c1=chain1.add(c1, 0);

      low=__shfl_down_sync(sync, current[0], 1, TPI);

      chain_t<> chain2;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index++)
        current[index]=chain2.madhi(t, x[index], current[index+1]);
      current[LIMBS-1]=chain2.madhi(t, x[LIMBS-1], c1);
      c1=chain2.add(0, 0);

      // could use the add3 trick
      current[LIMBS-1]=add_cc(current[LIMBS-1], low);
      c1=addc(c1, 0);
    }
  }
  
  mpset<LIMBS>(r, inverse);
  clear_padding(r);
}

} /* namespace cgbn */