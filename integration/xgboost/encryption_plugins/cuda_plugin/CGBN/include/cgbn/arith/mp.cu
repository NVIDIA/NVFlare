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

template<uint32_t limbs>
__device__ __forceinline__ void mpzero(uint32_t r[limbs]) {
  #pragma unroll
  for(int32_t index=0;index<limbs;index++)
    r[index]=0;
}

template<uint32_t limbs>
__device__ __forceinline__ void mpset(uint32_t r[limbs], const uint32_t a[limbs]) {
  #pragma unroll
  for(int32_t index=0;index<limbs;index++)
    r[index]=a[index];
}

template<uint32_t limbs>
__device__ __forceinline__ void mpswap(uint32_t x[limbs], uint32_t y[limbs]) {
  #pragma unroll
  for(int32_t index=0;index<limbs;index++) {
    uint32_t swap;
    
    swap=x[index];
    x[index]=y[index];
    y[index]=swap;
  }
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mplor(const uint32_t a[limbs]) {
  uint32_t r=a[0];
  
  #pragma unroll
  for(int32_t index=1;index<limbs;index++)
    r=r | a[index];
  return r;
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mpland(const uint32_t a[limbs]) {
  uint32_t r=a[0];
  
  #pragma unroll
  for(int32_t index=1;index<limbs;index++)
    r=r & a[index];
  return r;
}

template<uint32_t limbs>
__device__ __forceinline__ bool mpzeros(const uint32_t a[limbs]) {
  return mplor<limbs>(a)==0;
}

template<uint32_t limbs>
__device__ __forceinline__ bool mpones(const uint32_t a[limbs]) {
  return mpland<limbs>(a)==0xFFFFFFFF;
}

template<uint32_t limbs>
__device__ __forceinline__ void mpadd32_cc(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b) {
  chain_t<limbs,false,true> chain;
  r[0]=chain.add(a[0], b);
  #pragma unroll
  for(int32_t index=1;index<limbs;index++)
    r[index]=chain.add(a[index], 0);
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mpadd32(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b) {
  mpadd32_cc<limbs>(r, a, b);
  return addc(0, 0);
}


template<uint32_t limbs>
__device__ __forceinline__ void mpadd_cc(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs]) {
  chain_t<limbs,false,true> chain;
  #pragma unroll
  for(int32_t index=0;index<limbs;index++)
    r[index]=chain.add(a[index], b[index]);
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mpadd(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs]) {
  mpadd_cc<limbs>(r, a, b);
  return addc(0, 0);
}

template<uint32_t limbs>
__device__ __forceinline__ void mpsub32_cc(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b) {
  chain_t<limbs,false,true> chain;
  r[0]=chain.sub(a[0], b);
  #pragma unroll
  for(int32_t index=1;index<limbs;index++)
    r[index]=chain.sub(a[index], 0);
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mpsub32(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b) {
  mpsub32_cc<limbs>(r, a, b);
  return subc(0, 0);
}

template<uint32_t limbs>
__device__ __forceinline__ void mpsub_cc(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs]) {
  chain_t<limbs,false,true> chain;
  #pragma unroll
  for(int32_t index=0;index<limbs;index++)
    r[index]=chain.sub(a[index], b[index]);
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mpsub(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b[limbs]) {
  mpsub_cc<limbs>(r, a, b);
  return subc(0, 0);
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mpmul32(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t b) {
  uint32_t carry=0;
  
  #pragma unroll
  for(int32_t index=0;index<limbs;index++) {
    uint32_t temp=a[index];
    
    r[index]=madlo_cc(temp, b, carry);
    carry=madhic(temp, b, 0);
  }
  return carry;
}

/****************************************************************
 * requires that d is normalized
 ****************************************************************/
template<uint32_t limbs>
__device__ __forceinline__ uint32_t mprem32(const uint32_t a[limbs], const uint32_t d, const uint32_t approx) {
  uint32_t c, x0, x1, x2;
  
  // fast and avoids division
  if(limbs==1)
    return (a[0]>=d) ? a[0]-d : a[0];

  // set c to 2^64 % d
  c=-d*approx+d;
  
  x1=a[limbs-2];
  x2=a[limbs-1];
  #pragma unroll
  for(int32_t index=limbs-1;index>=2;index--) {
    x0=madlo_cc(x2, c, a[index-2]);
    x1=madhic_cc(x2, c, x1);
    x2=addc(0, 0);

    x0=madlo_cc(x2, c, x0);
    x1=madhic(x2, c, x1);

    x2=x1;
    x1=x0;
  }
  x2=(x2>=d) ? x2-d : x2;
  return urem(x1, x2, d, approx);  
}

template<uint32_t limbs>
__device__ __forceinline__ void mpleft(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t numbits, const uint32_t fill=0) {
  #pragma unroll
  for(int32_t index=limbs-1;index>=1;index--)
    r[index]=uleft_clamp(a[index-1], a[index], numbits);
  r[0]=uleft_clamp(fill, a[0], numbits);
}

template<uint32_t limbs>
__device__ __forceinline__ void mpright(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t numbits, const uint32_t fill=0) {
  #pragma unroll
  for(int32_t index=0;index<limbs-1;index++) 
    r[index]=uright_clamp(a[index], a[index+1], numbits);
  r[limbs-1]=uright_clamp(a[limbs-1], fill, numbits);
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mpclz(const uint32_t a[limbs]) {
  uint32_t word=0, count=0;
  
  #pragma unroll
  for(int32_t index=limbs-1;index>=0;index--) {
    word=(word!=0) ? word : a[index];
    count=(word!=0) ? count : (limbs-index)*32;
  }
  if(word!=0)
    count=count+uclz(word);
  return count;
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mpctz(const uint32_t a[limbs]) {
  uint32_t word=0, count=0;
  
  #pragma unroll
  for(int32_t index=0;index<limbs;index++) {
    word=(word!=0) ? word : a[index];
    count=(word!=0) ? count : index*32+32;
  }
  if(word!=0)
    count=count+uctz(word);
  return count;
}

template<uint32_t limbs>
__device__ __forceinline__ void mpmul(uint32_t lo[limbs], uint32_t hi[limbs], const uint32_t a[limbs], const uint32_t b[limbs]) {
  uint32_t c;
  
  #pragma unroll
  for(int32_t index=0;index<limbs;index++) {
    lo[index]=0;
    hi[index]=0;
  }
  
  #pragma unroll
  for(int32_t i=0;i<limbs;i++) {
    chain_t<limbs,false,true> chain1;
    #pragma unroll
    for(int32_t j=0;j<limbs;j++) {
      if(i+j<limbs)
        lo[i+j]=chain1.madlo(a[i], b[j], lo[i+j]);
      else
        hi[i+j-limbs]=chain1.madlo(a[i], b[j], hi[i+j-limbs]);
    }
    if(i==0)
      c=0;
    else
      c=addc(0, 0);
      
    chain_t<limbs> chain2;
    #pragma unroll
    for(int32_t j=0;j<limbs-1;j++) {
      if(i+j+1<limbs)
        lo[i+j+1]=chain2.madhi(a[i], b[j], lo[i+j+1]);
      else
        hi[i+j+1-limbs]=chain2.madhi(a[i], b[j], hi[i+j+1-limbs]);
    }
    hi[i]=chain2.madhi(a[i], b[limbs-1], c);
  }
}

template<uint32_t limbs, uint32_t max_rotation>
__device__ __forceinline__ void mprotate_left(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t numlimbs) {
  #pragma unroll
  for(int32_t index=0;index<limbs;index++) 
    r[index]=a[index];
  
  if(limbs>bit_set<max_rotation>::high_bit*2)
    shifter_t<limbs, bit_set<max_rotation>::high_bit, true>::mprotate_left(r, numlimbs);
  else if((limbs-1&limbs)==0)
    shifter_t<limbs, limbs/2, false>::mprotate_left(r, numlimbs);
  else
    shifter_t<limbs, bit_set<limbs>::high_bit, false>::mprotate_left(r, numlimbs);
}

template<uint32_t limbs, uint32_t max_rotation>
__device__ __forceinline__ void mprotate_right(uint32_t r[limbs], const uint32_t a[limbs], const uint32_t numlimbs) {
  #pragma unroll
  for(int32_t index=0;index<limbs;index++) 
    r[index]=a[index];
    
  if(limbs>bit_set<max_rotation>::high_bit*2)
    shifter_t<limbs, bit_set<max_rotation>::high_bit, true>::mprotate_right(r, numlimbs);
  else if((limbs-1&limbs)==0)
    shifter_t<limbs, limbs/2, false>::mprotate_right(r, numlimbs);
  else
    shifter_t<limbs, bit_set<limbs>::high_bit, false>::mprotate_right(r, numlimbs);
}

} /* cgbn namespace */