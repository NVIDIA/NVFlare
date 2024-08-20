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

template<uint32_t value>
class bit_set {
  public:
  static const uint32_t high_bit=bit_set<value/2>::high_bit<<1;
  static const uint32_t low_bit=(value-1 & ~value)+1;
};

template<>
class bit_set<1> {
  public:
  static const uint32_t high_bit=1;
  static const uint32_t low_bit=1;
};

/* mp rotate */
template<uint32_t limbs, uint32_t bit, bool loop>
class shifter_t {
  public:
  __device__ __forceinline__ static void mprotate_left(uint32_t r[limbs], uint32_t numlimbs);
  __device__ __forceinline__ static void mprotate_right(uint32_t r[limbs], uint32_t numlimbs);
};

template<uint32_t limbs>
class shifter_t<limbs, 0, false> {
  public:
  __device__ __forceinline__ static void mprotate_left(uint32_t r[limbs], uint32_t numlimbs);
  __device__ __forceinline__ static void mprotate_right(uint32_t r[limbs], uint32_t numlimbs);
};


/* rotate right/left primitives */

template<uint32_t limbs, uint32_t bit>
__device__ __forceinline__ static void mprotate_left_static(uint32_t r[limbs]) {
  int32_t  from, to, offset, pow2gcd=min(bit_set<limbs>::low_bit, bit);
  uint32_t swap;

  #pragma unroll
  for(int32_t index=0;index<limbs;index++) {
    offset=pow2gcd*index/limbs;
    to=(bit*index+offset)%limbs;
    from=(to+bit)%limbs;

    if(bit*index%limbs==0)
      swap=r[to];
    if((index+1)*bit%limbs!=0)
      r[to]=r[from];
    else
      r[to]=swap;
  }
}

template<uint32_t limbs, uint32_t bit>
__device__ __forceinline__ static void mprotate_left_bitcheck(uint32_t r[limbs], const uint32_t numlimbs) {
  int32_t  from, to, offset, pow2gcd=min(bit_set<limbs>::low_bit, bit);
  uint32_t swap;

  #pragma unroll
  for(int32_t index=0;index<limbs;index++) {
    offset=pow2gcd*index/limbs;
    to=(bit*index+offset)%limbs;
    from=(to+bit)%limbs;

    if(bit*index%limbs==0)
      swap=r[to];
    if((index+1)*bit%limbs!=0)
      r[to]=((bit & numlimbs)!=0) ? r[from] : r[to];
    else
      r[to]=((bit & numlimbs)!=0) ? swap : r[to];
  }
}

template<uint32_t limbs, uint32_t bit>
__device__ __forceinline__ static void mprotate_right_static(uint32_t r[limbs]) {
  int32_t  from, to, offset, pow2gcd=min(bit_set<limbs>::low_bit, bit);
  uint32_t swap;

  #pragma unroll
  for(int32_t index=limbs-1;index>=0;index--) {
    offset=pow2gcd*index/limbs;
    to=(bit*index+offset)%limbs;
    from=(to+limbs-bit)%limbs;

    if((index+1)*bit%limbs==0)
      swap=r[to];
    if(index*bit%limbs!=0)
      r[to]=r[from];
    else
      r[to]=swap;
  }
}

template<uint32_t limbs, uint32_t bit>
__device__ __forceinline__ static void mprotate_right_bitcheck(uint32_t r[limbs], const uint32_t numlimbs) {
  int32_t  from, to, offset, pow2gcd=min(bit_set<limbs>::low_bit, bit);
  uint32_t swap;

  #pragma unroll
  for(int32_t index=limbs-1;index>=0;index--) {
    offset=pow2gcd*index/limbs;
    to=(bit*index+offset)%limbs;
    from=(to+limbs-bit)%limbs;

    if((index+1)*bit%limbs==0)
      swap=r[to];
    if(index*bit%limbs!=0)
      r[to]=((bit & numlimbs)!=0) ? r[from] : r[to];
    else
      r[to]=((bit & numlimbs)!=0) ? swap : r[to];
  }
}


/* shifter implementation */

template<uint32_t limbs, uint32_t bit, bool loop>
__device__ __forceinline__ void shifter_t<limbs, bit, loop>::mprotate_left(uint32_t r[limbs], const uint32_t numlimbs) {
  if(loop) {
    uint32_t count=numlimbs;
    
    while(count>=bit) {
      mprotate_right_static<limbs, bit>(r);
      count-=bit;
    }
  }
  else 
    mprotate_right_bitcheck<limbs, bit>(r, numlimbs);
  shifter_t<limbs, bit/2, false>::mprotate_left(r, numlimbs);
}

template<uint32_t limbs>
__device__ __forceinline__ void shifter_t<limbs, 0, false>::mprotate_left(uint32_t r[limbs], const uint32_t numlimbs) {
}

template<uint32_t limbs, uint32_t bit, bool loop> 
__device__ __forceinline__ void shifter_t<limbs, bit, loop>::mprotate_right(uint32_t r[limbs], const uint32_t numlimbs) {
  if(loop) {
    uint32_t count=numlimbs;
    
    while(count>=bit) {
      mprotate_left_static<limbs, bit>(r);
      count-=bit;
    }
  }
  else 
    mprotate_left_bitcheck<limbs, bit>(r, numlimbs);
  shifter_t<limbs, bit/2, false>::mprotate_right(r, numlimbs);
}

template<uint32_t limbs>
__device__ __forceinline__ void shifter_t<limbs, 0, false>::mprotate_right(uint32_t r[limbs], const uint32_t numlimbs) {
}

} /* namespace cgbn */