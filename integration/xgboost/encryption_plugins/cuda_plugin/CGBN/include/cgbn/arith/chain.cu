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

#include <cstdint>

namespace cgbn {

template<uint32_t length, bool carry_in, bool carry_out>
__device__ __forceinline__ chain_t<length, carry_in, carry_out>::chain_t() : _position(0) {
}

template<uint32_t length, bool carry_in, bool carry_out>
__device__ __forceinline__ chain_t<length, carry_in, carry_out>::~chain_t() {
  if(_position<length && length!=0xFFFFFFFF) ASM_ERROR("chain destructor: chain too short / loop not unrolled");
  if(_position>length) ASM_ERROR("chain destructor: chain too long");
}

template<uint32_t length, bool carry_in, bool carry_out>
__device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::add(uint32_t a, uint32_t b) {
  uint32_t r;
  
  _position++;
  if(length==1 && _position==1 && !carry_in && !carry_out)
    r=a+b;
  else if(_position==1 && !carry_in)
    r=add_cc(a, b);
  else if(_position<length || carry_out)
    r=addc_cc(a, b);
  else
    r=addc(a, b);
  return r;
}

template<uint32_t length, bool carry_in, bool carry_out>
__device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::sub(uint32_t a, uint32_t b) {
  uint32_t r;
  
  _position++;
  if(length==1 && _position==1 && !carry_in && !carry_out)
    r=a-b;
  else if(_position==1 && !carry_in)
    r=sub_cc(a, b);
  else if(_position<length || carry_out)
    r=subc_cc(a, b);
  else
    r=subc(a, b);
  return r;
}

template<uint32_t length, bool carry_in, bool carry_out>
__device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::madlo(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  
  _position++;
  if(length==1 && _position==1 && !carry_in && !carry_out)
    r=madlo(a, b, c);
  else if(_position==1 && !carry_in)
    r=madlo_cc(a, b, c);
  else if(_position<length || carry_out)
    r=madloc_cc(a, b, c);
  else
    r=madloc(a, b, c);
  return r;
}

template<uint32_t length, bool carry_in, bool carry_out>
__device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::madhi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  
  _position++;
  if(length==1 && _position==1 && !carry_in && !carry_out)
    r=madhi(a, b, c);
  else if(_position==1 && !carry_in)
    r=madhi_cc(a, b, c);
  else if(_position<length || carry_out)
    r=madhic_cc(a, b, c);
  else
    r=madhic(a, b, c);
  return r;
}

template<uint32_t length, bool carry_in, bool carry_out>
__device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::xmadll(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  
  _position++;
  if(length==1 && _position==1 && !carry_in && !carry_out)
    r=xmadll(a, b, c);
  else if(_position==1 && !carry_in)
    r=xmadll_cc(a, b, c);
  else if(_position<length || carry_out)
    r=xmadllc_cc(a, b, c);
  else
    r=xmadllc(a, b, c);
  return r;
}

template<uint32_t length, bool carry_in, bool carry_out>
__device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::xmadlh(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  
  _position++;
  if(length==1 && _position==1 && !carry_in && !carry_out)
    r=xmadlh(a, b, c);
  else if(_position==1 && !carry_in)
    r=xmadlh_cc(a, b, c);
  else if(_position<length || carry_out)
    r=xmadlhc_cc(a, b, c);
  else
    r=xmadlhc(a, b, c);
  return r;
}

template<uint32_t length, bool carry_in, bool carry_out>
__device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::xmadhl(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  
  _position++;
  if(length==1 && _position==1 && !carry_in && !carry_out)
    r=xmadhl(a, b, c);
  else if(_position==1 && !carry_in)
    r=xmadhl_cc(a, b, c);
  else if(_position<length || carry_out)
    r=xmadhlc_cc(a, b, c);
  else
    r=xmadhlc(a, b, c);
  return r;
}

template<uint32_t length, bool carry_in, bool carry_out>
__device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::xmadhh(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  
  _position++;
  if(length==1 && _position==1 && !carry_in && !carry_out)
    r=xmadhh(a, b, c);
  else if(_position==1 && !carry_in)
    r=xmadhh_cc(a, b, c);
  else if(_position<length || carry_out)
    r=xmadhhc_cc(a, b, c);
  else
    r=xmadhhc(a, b, c);
  return r;
}

}
