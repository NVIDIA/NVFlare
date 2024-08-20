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
__device__ __forceinline__ uint32_t core_t<env>::extract_bits_ui32(const uint32_t a[LIMBS], const uint32_t start, const uint32_t len) {
  uint32_t sync=sync_mask();
  uint32_t offset, limb, lane, low, high, mask;
  
  if(start>=BITS)
    return 0;

  offset=start & 0x1F;
  limb=start>>5;
  mask=uleft_clamp(0xFFFFFFFF, 0, len);

  if(LIMBS==1) {
    low=__shfl_sync(sync, a[0], limb, TPI);
    if(BITS==32) 
      high=0;
    else if(offset+len<=32 || (int32_t)start>=(int32_t)(BITS-32))
      high=0;
    else
      high=__shfl_sync(sync, a[0], limb+1, TPI);
  }
  else {
    lane=static_divide_small<LIMBS>(limb);
    limb=limb-lane*LIMBS;

    low=0;
    high=0;
    #pragma unroll
    for(int index=0;index<LIMBS;index++) {
      low=(index==limb) ? a[index] : low;
      high=(index==limb) ? a[(index+1) % LIMBS] : high;
    }

    low=__shfl_sync(sync, low, lane, TPI);
    if(BITS==32)
      high=0;
    else if(offset+len<=32 || (int32_t)start>=(int32_t)(BITS-32))
      high=0;
    else
      high=__shfl_sync(sync, high, lane+(limb==LIMBS-1), TPI);
  }
  return uright_wrap(low, high, offset) & mask;
}

template<class env> 
__device__ __forceinline__ void core_t<env>::insert_bits_ui32(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t start, const uint32_t len, const uint32_t value) {
  int32_t  group_thread=threadIdx.x & TPI-1;
  uint32_t offset, limb, mask, data, shifted_mask_l, shifted_data_l, shifted_mask_h, shifted_data_h;

  if(PADDING!=0)
    if(start>=BITS)
      return;
  
  offset=start & 0x1F;
  limb=start>>5;
  mask=uleft_clamp(0xFFFFFFFF, 0, len);
  data=value & mask;

  if(LIMBS==1) {
    r[0]=a[0];
    if(group_thread==limb) {
      shifted_mask_l=~(mask<<offset);
      shifted_data_l=data<<offset;
      r[0] = (r[0] & shifted_mask_l) | shifted_data_l;
    }
    if(offset+len>32 && group_thread==limb+1) {
      shifted_mask_h=~uleft_wrap(mask, 0, offset);
      shifted_data_h=uleft_wrap(data, 0, offset);
      r[0] = (r[0] & shifted_mask_h) | shifted_data_h;
    }
  }
  else {
    limb=limb-group_thread*LIMBS;
    
    shifted_mask_l=~(mask<<offset);
    shifted_data_l=data<<offset;
    shifted_mask_h=~uleft_wrap(mask, 0, offset);
    shifted_data_h=uleft_wrap(data, 0, offset);
        
    if(offset+len<=32) {
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        r[index]=a[index];
        if(limb==index)
          r[index]=(r[index] & shifted_mask_l) | shifted_data_l;
      } 
    }
    else {
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        r[index]=a[index];
        if(limb==index)
          r[index]=(r[index] & shifted_mask_l) | shifted_data_l;
        if(limb+1==index)
          r[index]=(r[index] & shifted_mask_h) | shifted_data_h;
      }
    }
  }
  
  // we can write into the carry word
  clear_carry(r);
}

} /* namespace cgbn */