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
struct dispatch_padding_t {
  public:
  static const uint32_t TPI=core::TPI;
  static const uint32_t LIMBS=core::LIMBS;
  static const uint32_t BITS=core::BITS;
  
  static const uint32_t PAD_THREAD=core::PAD_THREAD;
  static const uint32_t PAD_LIMB=core::PAD_LIMB;

  __device__ __forceinline__ static uint32_t clear_carry(uint32_t &x) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI-1;
    uint32_t result;
    
    result=__shfl_sync(sync, x, PAD_THREAD, TPI);
    x=(group_thread<PAD_THREAD) ? x : 0;
    return result;
  }
  
  __device__ __forceinline__ static uint32_t clear_carry(uint32_t x[LIMBS]) {
    uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI-1;
    uint32_t result;
    
    result=__shfl_sync(sync, x[PAD_LIMB], PAD_THREAD, TPI);
    x[PAD_LIMB]=(group_thread!=PAD_THREAD) ? x[PAD_LIMB] : 0;
    return result;
  }

  __device__ __forceinline__ static void clear_padding(uint32_t &x) {
    uint32_t group_thread=threadIdx.x & TPI-1;

    x=(group_thread<PAD_THREAD) ? x : 0;
  }

  __device__ __forceinline__ static void clear_padding(uint32_t x[LIMBS]) {
    uint32_t group_thread=threadIdx.x & TPI-1;
    int32_t  group_base=group_thread*LIMBS;
    
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      x[index]=(group_base<BITS/32-index) ? x[index] : 0;
  }
};

template<class core>
struct dispatch_padding_t<core, 0> {
  public:
  static const uint32_t LIMBS=core::LIMBS;

  __device__ __forceinline__ static uint32_t clear_carry(uint32_t &x) {
    return 0;
  }

  __device__ __forceinline__ static uint32_t clear_carry(uint32_t x[LIMBS]) {
    return 0;
  }

  __device__ __forceinline__ static void clear_padding(uint32_t &x) {
  }

  __device__ __forceinline__ static void clear_padding(uint32_t x[LIMBS]) {
  }
};

} /* namespace cgbn */
