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
class dispatch_masking_t {
  public:
  static const uint32_t LIMBS=core::LIMBS;
  static const uint32_t TPI=core::TPI;
  
  static const int32_t  bits=(int32_t)core::BITS;
  
  __device__ __forceinline__ static void bitwise_mask_copy(uint32_t r[LIMBS], const int32_t numbits) {
    int32_t group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
    
    if(numbits>=bits || numbits<=-bits) {
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        r[index]=(group_base<bits/32-index) ? 0xFFFFFFFF : 0;
    }
    else if(numbits>=0) {
      int32_t limb=(numbits>>5)-group_base;
      int32_t straddle=uleft_wrap(0xFFFFFFFF, 0, numbits);

      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        if(limb<index)
          r[index]=0;
        else if(limb>index)
          r[index]=0xFFFFFFFF;
        else
          r[index]=straddle;
      }
    }
    else {
      int32_t limb=(numbits+bits>>5)-group_base;
      int32_t straddle=uleft_wrap(0, 0xFFFFFFFF, numbits);
  
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        if(limb>index || group_base>=bits/32-index)
          r[index]=0;
        else if(limb<index)
          r[index]=0xFFFFFFFF;
        else
          r[index]=straddle;
      }
    }
  }
  
  __device__ __forceinline__ static void bitwise_mask_and(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    int32_t group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
    
    if(numbits>=bits || numbits<=-bits) {
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        r[index]=a[index];
    }
    else if(numbits>=0) {
      int32_t limb=(numbits>>5)-group_base;
      int32_t straddle=uleft_wrap(0xFFFFFFFF, 0, numbits);

      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        if(limb<index)
          r[index]=0;
        else if(limb>index)
          r[index]=a[index];
        else
          r[index]=a[index] & straddle;
      }
    }
    else {
      int32_t limb=(numbits+bits>>5)-group_base;
      int32_t straddle=uleft_wrap(0, 0xFFFFFFFF, numbits);
  
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        if(limb>index)
          r[index]=0;
        else if(limb<index)
          r[index]=a[index];
        else
          r[index]=a[index] & straddle;
      }
    }
  }

  __device__ __forceinline__ static void bitwise_mask_ior(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    int32_t group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
    
    if(numbits>=bits || numbits<=-bits) {
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        r[index]=(group_base<bits/32-index) ? 0xFFFFFFFF : 0;
    }
    else if(numbits>=0) {
      int32_t limb=(numbits>>5)-group_base;
      int32_t straddle=uleft_wrap(0xFFFFFFFF, 0, numbits);

      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        if(limb<index)
          r[index]=a[index];
        else if(limb>index)
          r[index]=0xFFFFFFFF;
        else
          r[index]=a[index] | straddle;
      }
    }
    else {
      int32_t limb=(numbits+bits>>5)-group_base;
      int32_t straddle=uleft_wrap(0, 0xFFFFFFFF, numbits);
  
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        if(limb>index || group_base>=bits/32-index)
          r[index]=a[index];
        else if(limb<index)
          r[index]=0xFFFFFFFF;
        else
          r[index]=a[index] | straddle;
      }
    }
  }

  __device__ __forceinline__ static void bitwise_mask_xor(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    int32_t group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
    
    if(numbits>=bits || numbits<=-bits) {
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        r[index]=(group_base<bits/32-index) ? a[index] ^ 0xFFFFFFFF : 0;
    }
    else if(numbits>=0) {
      int32_t limb=(numbits>>5)-group_base;
      int32_t straddle=uleft_wrap(0xFFFFFFFF, 0, numbits);

      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        if(limb<index)
          r[index]=a[index];
        else if(limb>index)
          r[index]=a[index] ^ 0xFFFFFFFF;
        else
          r[index]=a[index] ^ straddle;
      }
    }
    else {
      int32_t limb=(numbits+bits>>5)-group_base;
      int32_t straddle=uleft_wrap(0, 0xFFFFFFFF, numbits);
  
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        if(limb>index || group_base>=bits/32-index)
          r[index]=a[index];
        else if(limb<index)
          r[index]=a[index] ^ 0xFFFFFFFF;
        else
          r[index]=a[index] ^ straddle;
      }
    }
  }  

  __device__ __forceinline__ static void bitwise_mask_select(uint32_t r[LIMBS], const uint32_t clear[LIMBS], const uint32_t set[LIMBS], const int32_t numbits) {
    int32_t group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
  
    if(numbits>=bits || numbits<=-bits) {
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) 
        r[index]=set[index];
    }
    else if(numbits>=0) {
      int32_t limb=(numbits>>5)-group_base;
      int32_t straddle=uleft_wrap(0xFFFFFFFF, 0, numbits);
  
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        if(limb<index)
          r[index]=clear[index];
        else if(limb>index)
          r[index]=set[index];
        else
          r[index]=(set[index] & straddle) | (clear[index] & ~straddle);
      }
    }
    else {
      int32_t limb=(bits+numbits>>5)-group_base;
      int32_t straddle=uleft_wrap(0, 0xFFFFFFFF, numbits);
  
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        if(limb<index)
          r[index]=set[index];
        else if(limb>index)
          r[index]=clear[index];
        else
          r[index]=(set[index] & straddle) | (clear[index] & ~straddle);
      }
    }
  }
};

template<class core>
class dispatch_masking_t<core, 0> {
  public:
  static const uint32_t LIMBS=core::LIMBS;
  static const uint32_t TPI=core::TPI;
    
  __device__ __forceinline__ static void bitwise_mask_copy(uint32_t r[LIMBS], const int32_t numbits) {
    dmask_set<TPI, LIMBS>(r, numbits);
  }

  __device__ __forceinline__ static void bitwise_mask_and(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    dmask_and<TPI, LIMBS>(r, a, numbits);
  }

  __device__ __forceinline__ static void bitwise_mask_ior(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    dmask_ior<TPI, LIMBS>(r, a, numbits);
  }

  __device__ __forceinline__ static void bitwise_mask_xor(uint32_t r[LIMBS], const uint32_t a[LIMBS], const int32_t numbits) {
    dmask_xor<TPI, LIMBS>(r, a, numbits);
  }

  __device__ __forceinline__ static void bitwise_mask_select(uint32_t r[LIMBS], const uint32_t clear[LIMBS], const uint32_t set[LIMBS], const int32_t numbits) {
    dmask_select<TPI, LIMBS>(r, clear, set, numbits);
  }
};

} /* namespace cgbn */
