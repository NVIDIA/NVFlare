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

template<uint32_t tpi, uint32_t limbs>
__device__ __forceinline__ bool dequals(const uint32_t sync, const uint32_t a[limbs], const uint32_t b[limbs]) {
  static const uint32_t TPI_ONES=(1ull<<tpi)-1;
  
  uint32_t group_thread=threadIdx.x & tpi-1, warp_thread=threadIdx.x & warpSize-1;
  uint32_t lor, mask;
  
  lor=a[0] ^ b[0];
  #pragma unroll
  for(int32_t index=1;index<limbs;index++)
    lor=lor | (a[index] ^ b[index]);
  mask=__ballot_sync(sync, lor==0);
  if(tpi<warpSize)
    mask=mask>>(group_thread ^ warp_thread);
  return mask==TPI_ONES;
}

template<uint32_t tpi, uint32_t limbs>
__device__ __forceinline__ int32_t dcompare(const uint32_t sync, const uint32_t a[limbs], const uint32_t b[limbs]) {
  static const uint32_t TPI_ONES=(1ull<<tpi)-1;
  
  uint32_t group_thread=threadIdx.x & tpi-1, warp_thread=threadIdx.x & warpSize-1;
  uint32_t a_ballot, b_ballot;

  if(limbs==1) {
    a_ballot=__ballot_sync(sync, a[0]>=b[0]);
    b_ballot=__ballot_sync(sync, a[0]<=b[0]);
  }
  else {
    chain_t<> chain1;
    #pragma unroll
    for(int32_t index=0;index<limbs;index++)
      chain1.sub(a[index], b[index]);
    a_ballot=chain1.sub(0, 0);
    a_ballot=__ballot_sync(sync, a_ballot==0);
    
    chain_t<> chain2;
    #pragma unroll
    for(int32_t index=0;index<limbs;index++)
      chain2.sub(b[index], a[index]);
    b_ballot=chain2.sub(0, 0);
    b_ballot=__ballot_sync(sync, b_ballot==0);
  }
  
  if(tpi<warpSize) {
    uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);
    
    a_ballot=a_ballot & mask;
    b_ballot=b_ballot & mask;
  }
  
  return ucmp(a_ballot, b_ballot);
}

template<uint32_t tpi, uint32_t limbs>
__device__ __forceinline__ void dmask_set(uint32_t r[limbs], const int32_t numbits) {
  int32_t group_thread=threadIdx.x & tpi-1, group_base=group_thread*limbs;
  int32_t bits=tpi*limbs*32;
  
  if(numbits>=bits || numbits<=-bits) {
    #pragma unroll
    for(int32_t index=0;index<limbs;index++)
      r[index]=0xFFFFFFFF;
  }
  else if(numbits>=0) {
    int32_t limb=(numbits>>5)-group_base;
    int32_t straddle=uleft_wrap(0xFFFFFFFF, 0, numbits);

    #pragma unroll
    for(int32_t index=0;index<limbs;index++) {
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
    for(int32_t index=0;index<limbs;index++) {
      if(limb<index)
        r[index]=0xFFFFFFFF;
      else if(limb>index)
        r[index]=0;
      else
        r[index]=straddle;
    }
  }
}

template<uint32_t tpi, uint32_t limbs>
__device__ __forceinline__ void dmask_and(uint32_t r[limbs], const uint32_t a[limbs], const int32_t numbits) {
  int32_t group_thread=threadIdx.x & tpi-1, group_base=group_thread*limbs;
  int32_t bits=tpi*limbs*32;
  
  if(numbits>=bits || numbits<=-bits) {
    #pragma unroll
    for(int32_t index=0;index<limbs;index++)
      r[index]=a[index];
  }
  else if(numbits>=0) {
    int32_t limb=(numbits>>5)-group_base;
    int32_t straddle=uleft_wrap(0xFFFFFFFF, 0, numbits);

    #pragma unroll
    for(int32_t index=0;index<limbs;index++) {
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
    for(int32_t index=0;index<limbs;index++) {
      if(limb<index)
        r[index]=a[index];
      else if(limb>index)
        r[index]=0;
      else
        r[index]=a[index] & straddle;
    }
  }
}

template<uint32_t tpi, uint32_t limbs>
__device__ __forceinline__ void dmask_ior(uint32_t r[limbs], const uint32_t a[limbs], const int32_t numbits) {
  int32_t group_thread=threadIdx.x & tpi-1, group_base=group_thread*limbs;
  int32_t bits=tpi*limbs*32;
  
  if(numbits>=bits || numbits<=-bits) {
    #pragma unroll
    for(int32_t index=0;index<limbs;index++)
      r[index]=0xFFFFFFFF;
  }
  else if(numbits>=0) {
    int32_t limb=(numbits>>5)-group_base;
    int32_t straddle=uleft_wrap(0xFFFFFFFF, 0, numbits);

    #pragma unroll
    for(int32_t index=0;index<limbs;index++) {
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
    for(int32_t index=0;index<limbs;index++) {
      if(limb<index)
        r[index]=0xFFFFFFFF;
      else if(limb>index)
        r[index]=a[index];
      else
        r[index]=a[index] | straddle;
    }
  }
}

template<uint32_t tpi, uint32_t limbs>
__device__ __forceinline__ void dmask_xor(uint32_t r[limbs], const uint32_t a[limbs], const int32_t numbits) {
  int32_t group_thread=threadIdx.x & tpi-1, group_base=group_thread*limbs;
  int32_t bits=tpi*limbs*32;
  
  if(numbits>=bits || numbits<=-bits) {
    #pragma unroll
    for(int32_t index=0;index<limbs;index++)
      r[index]=a[index] ^ 0xFFFFFFFF;
  }
  else if(numbits>=0) {
    int32_t limb=(numbits>>5)-group_base;
    int32_t straddle=uleft_wrap(0xFFFFFFFF, 0, numbits);

    #pragma unroll
    for(int32_t index=0;index<limbs;index++) {
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
    for(int32_t index=0;index<limbs;index++) {
      if(limb<index)
        r[index]=a[index] ^ 0xFFFFFFFF;
      else if(limb>index)
        r[index]=a[index];
      else
        r[index]=a[index] ^ straddle;
    }
  }
}

template<uint32_t tpi, uint32_t limbs>
__device__ __forceinline__ void dmask_select(uint32_t r[limbs], const uint32_t clear[limbs], const uint32_t set[limbs], int32_t numbits) {
  int32_t group_thread=threadIdx.x & tpi-1, group_base=group_thread*limbs;
  int32_t bits=tpi*limbs*32;

  if(numbits>=bits || numbits<=-bits) {
    #pragma unroll
    for(int32_t index=0;index<limbs;index++) 
      r[index]=set[index];
  }
  else if(numbits>=0) {
    int32_t limb=(numbits>>5)-group_base;
    int32_t straddle=uleft_wrap(0xFFFFFFFF, 0, numbits);

    #pragma unroll
    for(int32_t index=0;index<limbs;index++) {
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
    for(int32_t index=0;index<limbs;index++) {
      if(limb<index)
        r[index]=set[index];
      else if(limb>index)
        r[index]=clear[index];
      else
        r[index]=(set[index] & straddle) | (clear[index] & ~straddle);
    }
  }
}

template<uint32_t tpi, uint32_t limbs, uint32_t max_rotation>
__device__ __forceinline__ void drotate_left(const uint32_t sync, uint32_t r[limbs], const uint32_t x[limbs], const uint32_t numbits) {
  uint32_t rotate_bits=numbits & 0x1F, numlimbs=numbits>>5, threads=static_divide_small<limbs>(numlimbs);

  numlimbs=numlimbs-threads*limbs;
  if(numlimbs==0) {
    #pragma unroll
    for(int32_t index=0;index<limbs;index++)
      r[index]=__shfl_sync(sync, x[index], threadIdx.x-threads, tpi);
  }
  else {
    mprotate_left<limbs, max_rotation>(r, x, numlimbs);
    #pragma unroll
    for(int32_t index=0;index<limbs;index++)
       r[index]=__shfl_sync(sync, r[index], threadIdx.x-threads-(index<numlimbs), tpi);
  }

  if(rotate_bits>0) {
    uint32_t fill=__shfl_sync(sync, r[limbs-1], threadIdx.x-1, tpi);

    mpleft<limbs>(r, r, rotate_bits, fill);
  }
}

template<uint32_t tpi, uint32_t limbs, uint32_t max_rotation>
__device__ __forceinline__ void drotate_right(const uint32_t sync, uint32_t r[limbs], const uint32_t x[limbs], const uint32_t numbits) {
  uint32_t rotate_bits=numbits & 0x1F, numlimbs=numbits>>5, threads=static_divide_small<limbs>(numlimbs);

  numlimbs=numlimbs-threads*limbs;
  if(numlimbs==0) {
    #pragma unroll
    for(int32_t index=0;index<limbs;index++)
      r[index]=__shfl_sync(sync, x[index], threadIdx.x+threads, tpi);
  }
  else {
    mprotate_right<limbs, max_rotation>(r, x, numlimbs);
    #pragma unroll
    for(int32_t index=0;index<limbs;index++)
      r[index]=__shfl_sync(sync, r[index], threadIdx.x+threads+(limbs-index<=numlimbs), tpi);
  }

  if(rotate_bits>0) {
    uint32_t fill=__shfl_sync(sync, r[0], threadIdx.x+1, tpi);

    mpright<limbs>(r, r, rotate_bits, fill);
  }
}

template<uint32_t tpi, uint32_t limbs, bool zero>
__device__ __forceinline__ void dscatter(const uint32_t sync, uint32_t &dest, const uint32_t source[limbs], const uint32_t source_thread=31) {
  uint32_t group_thread=threadIdx.x & tpi-1;
  uint32_t t;
  
  if(zero)
    dest=0;
  #pragma unroll
  for(int32_t index=0;index<limbs;index++) {
    t=__shfl_sync(sync, source[index], source_thread, tpi);
    dest=(group_thread==tpi-limbs+index) ? t : dest;
  }  
}

template<uint32_t tpi, uint32_t limbs>
__device__ __forceinline__ void dall_gather(const uint32_t sync, uint32_t dest[limbs], const uint32_t source) {
  #pragma unroll
  for(int32_t index=0;index<limbs;index++) 
    dest[index]=__shfl_sync(sync, source, tpi-limbs+index, tpi);
}

template<uint32_t tpi, uint32_t limbs, bool zero>
__device__ __forceinline__ void fwgather(const uint32_t sync, uint32_t dest[limbs], const uint32_t source, const uint32_t destination_thread=31) {
  uint32_t group_thread=threadIdx.x & warpSize-1;
  uint32_t t;

  if(zero) {
    #pragma unroll
    for(int32_t index=0;index<limbs;index++) {
      t=__shfl_sync(sync, source, tpi-limbs+index, tpi);
      dest[index]=(group_thread==destination_thread) ? t : 0;
    }
  }
  else {
    #pragma unroll
    for(int32_t index=0;index<limbs;index++) {
      t=__shfl_sync(sync, source, tpi-limbs+index, tpi);
      dest[index]=(group_thread==destination_thread) ? t : dest[index];
    }
  }
}



} /* namespace cgbn */