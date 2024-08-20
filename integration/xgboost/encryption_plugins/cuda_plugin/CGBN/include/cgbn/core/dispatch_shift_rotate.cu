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

template<class core, uint32_t padding>
class dispatch_shift_rotate_t {
  public:
  static const uint32_t TPI=core::TPI;
  static const uint32_t LIMBS=core::LIMBS;
  static const uint32_t BITS=core::BITS;
  static const uint32_t MAX_ROTATION=core::MAX_ROTATION;
  
  __device__ __forceinline__ static void shift_left(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t numbits) {
    uint32_t group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
    int32_t  delta;
    
    if(numbits<BITS) {
      drotate_left<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
      
      delta=(numbits>>5)-group_base;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        if(group_base>=BITS/32-index || delta>index)
          r[index]=0;
    }
    else
      mpzero<LIMBS>(r);
  }
  
  __device__ __forceinline__ static void shift_right(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t numbits) {
    uint32_t group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
    int32_t  delta;

    if(numbits<BITS) {
      drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);

      delta=group_base-(BITS-numbits>>5);      
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        if(delta>index)
          r[index]=0;
    }
    else
      mpzero<LIMBS>(r);
  }
  
  __device__ __forceinline__ static void rotate_left(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t numbits) {
    uint32_t group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
    uint32_t amount=numbits%BITS, temp[LIMBS];

    drotate_left<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), temp, a, amount+TPI*LIMBS*32-BITS);
    drotate_left<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, amount);
    
    dmask_select<TPI, LIMBS>(r, r, temp, amount);

    // clear padding
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      if(group_base>=BITS/32-index)
        r[index]=0;
  }
  
  __device__ __forceinline__ static void rotate_right(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t numbits) {
    uint32_t group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
    uint32_t amount=numbits%BITS, temp[LIMBS];

    drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), temp, a, amount);
    drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, amount+TPI*LIMBS*32-BITS);
    
    dmask_select<TPI, LIMBS>(r, r, temp, BITS-amount);

    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      if(group_base>=BITS/32-index)
        r[index]=0;
  }  
};

template<class core>
class dispatch_shift_rotate_t<core, 0> {
  public:
  static const uint32_t TPI=core::TPI;
  static const uint32_t LIMBS=core::LIMBS;
  static const uint32_t BITS=core::BITS;
  static const uint32_t MAX_ROTATION=core::MAX_ROTATION;
  
  __device__ __forceinline__ static void shift_left(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t numbits) {
    if(numbits<BITS) {
      drotate_left<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
      dmask_and<TPI, LIMBS>(r, r, (int32_t)(numbits-BITS));
    }
    else
      mpzero<LIMBS>(r);
  }
  
  __device__ __forceinline__ static void shift_right(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t numbits) {
    if(numbits<BITS) {
      drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
      dmask_and<TPI, LIMBS>(r, r, BITS-numbits);
    }
    else
      mpzero<LIMBS>(r);
  }
  
  __device__ __forceinline__ static void rotate_left(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t numbits) {
    drotate_left<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
  }
  
  __device__ __forceinline__ static void rotate_right(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t numbits) {
    drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
  }
};

} /* namespace cgbn */