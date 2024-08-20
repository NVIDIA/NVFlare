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
__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t subc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t madlo(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t madloc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint64_t mad_wide(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r;
  
  asm volatile ("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadll(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %al, %bl;\n\t"
                "add.u32       %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadll_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %al, %bl;\n\t"
                "add.cc.u32    %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadllc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %al, %bl;\n\t"
                "addc.cc.u32   %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadllc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %al, %bl;\n\t"
                "addc.u32      %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadlh(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %al, %bh;\n\t"
                "add.u32       %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadlh_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %al, %bh;\n\t"
                "add.cc.u32    %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadlhc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %al, %bh;\n\t"
                "addc.cc.u32   %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadlhc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %al, %bh;\n\t"
                "addc.u32      %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadhl(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %ah, %bl;\n\t"
                "add.u32       %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadhl_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %ah, %bl;\n\t"
                "add.cc.u32    %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadhlc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %ah, %bl;\n\t"
                "addc.cc.u32   %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadhlc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %ah, %bl;\n\t"
                "addc.u32      %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadhh(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %ah, %bh;\n\t"
                "add.u32       %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadhh_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %ah, %bh;\n\t"
                "add.cc.u32    %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadhhc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %ah, %bh;\n\t"
                "addc.cc.u32   %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t xmadhhc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                "mov.b32       {%al,%ah},%1;\n\t"
                "mov.b32       {%bl,%bh},%2;\n\t"
                "mul.wide.u16  %0, %ah, %bh;\n\t"
                "addc.u32      %0, %0, %3;\n\t"
                "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t umin(uint32_t a, uint32_t b) {
  uint32_t r;
  
  asm volatile ("min.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t umax(uint32_t a, uint32_t b) {
  uint32_t r;
  
  asm volatile ("max.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t uleft_clamp(uint32_t lo, uint32_t hi, uint32_t amt) {
  uint32_t r;
  
  #if __CUDA_ARCH__>=320   
    asm volatile ("shf.l.clamp.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(amt));
  #else
    amt=umin(amt, 32);
    r=hi<<amt;
    r=r | (lo>>32-amt);
  #endif
  return r;
}

__device__ __forceinline__ uint32_t uright_clamp(uint32_t lo, uint32_t hi, uint32_t amt) {
  uint32_t r;
  
  #if __CUDA_ARCH__>=320   
    asm volatile ("shf.r.clamp.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(amt));
  #else
    amt=umin(amt, 32);
    r=lo>>amt;
    r=r | (hi<<32-amt);
  #endif
  return r;
}

__device__ __forceinline__ uint32_t uleft_wrap(uint32_t lo, uint32_t hi, uint32_t amt) {
  uint32_t r;
  
  #if __CUDA_ARCH__>=320   
    asm volatile ("shf.l.wrap.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(amt));
  #else
    amt=amt & 0x1F;
    r=hi<<amt;
    r=r | (lo>>32-amt);
  #endif
  return r;
}

__device__ __forceinline__ uint32_t uright_wrap(uint32_t lo, uint32_t hi, uint32_t amt) {
  uint32_t r;
  
  #if __CUDA_ARCH__>=320   
    asm volatile ("shf.r.wrap.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(amt));
  #else
    amt=amt & 0x1F;
    r=lo>>amt;
    r=r | (hi<<32-amt);
  #endif
  return r;
}

__device__ __forceinline__ uint32_t uabs(int32_t x) {
  uint32_t r;
  
  asm volatile ("abs.s32 %0,%1;" : "=r"(r) : "r"(x));
  return r;
}

__device__ __forceinline__ uint32_t uhigh(uint64_t wide) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u32 %ignore;\n\t"
                "mov.b64 {%ignore,%0},%1;\n\t"
                "}" : "=r"(r) : "l"(wide));
  return r;
}

__device__ __forceinline__ uint32_t ulow(uint64_t wide) {
  uint32_t r;

  asm volatile ("{\n\t"
                ".reg .u32 %ignore;\n\t"
                "mov.b64 {%0,%ignore},%1;\n\t"
                "}" : "=r"(r) : "l"(wide));
  return r;
}

__device__ __forceinline__ uint64_t make_wide(uint32_t lo, uint32_t hi) {
  uint64_t r;
  
  asm volatile ("mov.b64 %0,{%1,%2};" : "=l"(r) : "r"(lo), "r"(hi));
  return r;
}

} /* namespace cgbn */


