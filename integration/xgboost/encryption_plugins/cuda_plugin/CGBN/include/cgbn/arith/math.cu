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

__device__ __forceinline__ int32_t ushiftamt(uint32_t x) {
  uint32_t r;
  
  asm volatile ("bfind.shiftamt.u32 %0,%1;" : "=r"(r) : "r"(x));
  return r;
}

__device__ __forceinline__ int32_t ucmp(uint32_t a, uint32_t b) {
  int32_t compare;
  
  compare=(a>b) ? 1 : 0;
  compare=(a<b) ? -1 : compare;
  return compare;
}

__device__ __forceinline__ uint32_t uclz(uint32_t x) {
  return __clz(x);
}

__device__ __forceinline__ uint32_t uctz(uint32_t x) {
  return 31-ushiftamt(x-1^x);
}

__device__ __forceinline__ uint32_t ubinary_inverse(uint32_t x) {
  uint32_t inv=x;

  inv=inv*(inv*x+14);
  inv=inv*(inv*x+2);
  inv=inv*(inv*x+2);
  inv=inv*(inv*x+2);
  return -inv;
}

__device__ __forceinline__ uint32_t ugcd(uint32_t a, uint32_t b) {
  uint32_t acnt, bcnt, gcdcnt, t;

  if(a==0) return b;
  if(b==0) return a;
  
  acnt=uctz(a);
  bcnt=uctz(b);
  
  a=a>>acnt;
  b=b>>bcnt;
  gcdcnt=umin(acnt, bcnt);

  while(a!=b) {
    t=__usad(a, b, 0);
    b=umin(a, b);
    a=t>>1;
    while((a & 0x01)==0)
      a=a>>1;
  }
  return a<<gcdcnt;
}

/*
__device__ __forceinline__ uint32_t usqrt(const uint32_t x) {
  uint32_t   s=0, t, r, neg=~x;

  s=(x>=0x40000000) ? 0x8000 : 0;
  #pragma unroll
  for(int32_t bit=0x4000;bit>=0x1;bit=bit/2) {
    t=s+bit;
    r=t*t+neg;
    s=(0>(int32_t)r) ? t : s;
  }
  return s;
}
*/

/*
__device__ __forceinline__ uint32_t uapprox(const uint32_t d) {
  uint64_t a=d;

  if(d==0x80000000)
    return 0xFFFFFFFF;

  // computes ceil(2^64/d) - 2^32
  a=-a / d;
  return ((uint32_t)a)+2;
}
*/

/****************************************************************
 * d must be normalized (i.e., d>=0x80000000)
 * computes ceil(2^64/d)-2^32       if d>0x80000000
 *          ceil(2^64/d)-(2^32+1)   if d=0x80000000 
 ****************************************************************/
__device__ __forceinline__ uint32_t uapprox(uint32_t d) {
  float    f;
  uint32_t a, t0, t1;
  int32_t  s;

  // special case d=0x80000000
  if(d==0x80000000)
    return 0xFFFFFFFF;
  
  // get a first estimate using float 1/x
  f=__uint_as_float((d>>8) + 0x3F000000);
  asm volatile("rcp.approx.f32 %0,%1;" : "=f"(f) : "f"(f));
  a=__float_as_uint(f);
  a=madlo(a, 512, 0xFFFFFE00);
  
  // use Newton-Raphson to improve estimate
  s=madhi(d, a, d);
  t0=abs(s);
  t0=madhi(t0, a, t0);
  a=(s>=0) ? a-t0 : a+t0;

  // two corrections steps give exact result
  a=a-madhi(d, a, d);       // first correction

  t0=madlo_cc(d, a, 0);     // second correction
  t1=madhic(d, a, d);
  t1=(t1!=0) ? t1 : (t0>=d);
  a=a-t1;
  return a;
}

__device__ __forceinline__ uint32_t udiv(const uint32_t lo, const uint32_t hi, const uint32_t d, const uint32_t approx) {
  uint32_t q, add, ylo, yhi;

  // q=MIN(0xFFFFFFFF, HI(approx * hi) + hi + ((lo<d) ? 1 : 2));
  sub_cc(lo, d);
  add=subc(hi, 0xFFFFFFFE);
  q=madhi(hi, approx, add);
  q=(q<hi) ? 0xFFFFFFFF : q;           // the only case where this can carry out is if hi and approx are both 0xFFFFFFFF and add=2
                                       // but in this case, q will end up being 0xFFFFFFFF, which is what we want
                                       // if q+hi carried out, set q to 0xFFFFFFFF

  ylo=madlo(q, d, 0);
  yhi=madhi(q, d, 0);

  ylo=sub_cc(lo, ylo);      // first correction step
  yhi=subc_cc(hi, yhi);
  add=subc(0, 0);
  q=q+add;

  ylo=add_cc(ylo, d);       // second correction step
  yhi=addc_cc(yhi, 0);
  q=addc(q, add);

  return q;
}

/****************************************************************
 * requires that hi<d
 ****************************************************************/
__device__ __forceinline__ uint32_t urem(const uint32_t lo, const uint32_t hi, const uint32_t d, const uint32_t approx) {
  uint32_t q, add, ylo;
  int32_t  yhi;
  
  // q=MIN(0xFFFFFFFF, HI(approx * hi) + hi + ((lo<d) ? 1 : 2));
  sub_cc(lo, d);
  add=subc(hi, 0xFFFFFFFE);
  q=madhi(hi, approx, add);
  q=(q<hi) ? 0xFFFFFFFF : q;           // the only case where this can carry out is if hi and approx are both 0xFFFFFFFF and add=2
                                       // but in this case, q will end up being 0xFFFFFFFF, which is what we want
                                       // if q+hi carried out, set q to 0xFFFFFFFF

  ylo=madlo(q, d, 0);
  yhi=madhi(q, d, 0);

  ylo=sub_cc(lo, ylo);      // first correction step
  yhi=subc_cc(hi, yhi);
  add=(yhi<0) ? d : 0;
  
  ylo=add_cc(ylo, add);     // second correction step
  yhi=addc_cc(yhi, 0);
  add=(yhi<0) ? d : 0;

  ylo=add_cc(ylo, add);

  return ylo;
}

__device__ __forceinline__ uint32_t udiv(const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t d0, const uint32_t d1, const uint32_t approx) {
  uint32_t q, add, y0, y1, y2;

  // q=MIN(0xFFFFFFFF, HI(approx * hi) + hi + ((lo<d) ? 1 : 2));
  sub_cc(x1, d1);
  add=subc(x2, 0xFFFFFFFE);
  q=madhi(x2, approx, add);
  q=(q<x2) ? 0xFFFFFFFF : q;           // the only case where this can carry out is if hi and approx are both 0xFFFFFFFF and add=2
                                       // but in this case, q will end up being 0xFFFFFFFF, which is what we want
                                       // if q+hi carried out, set q to 0xFFFFFFFF

  y0=madlo(q, d0, 0);
  y1=madhi(q, d0, 0);
  y1=madlo_cc(q, d1, y1);
  y2=madhic(q, d1, 0);
  
  y0=sub_cc(x0, y0);    // first correction
  y1=subc_cc(x1, y1);
  y2=subc_cc(x2, y2);
  add=subc(0, 0);
  q=q+add;

  y0=add_cc(y0, d0);    // second correction
  y1=addc_cc(y1, d1);
  y2=addc_cc(y2, 0);
  add=addc(add, 0);
  q=q+add;
  
  y0=add_cc(y0, d0);    // third correction
  y1=addc_cc(y1, d1);
  y2=addc_cc(y2, 0);
  add=addc(add, 0);
  q=q+add;
  
  y0=add_cc(y0, d0);    // fourth correction
  y1=addc_cc(y1, d1);
  y2=addc_cc(y2, 0);
  q=addc(q, add);
  
  return q;
}

__device__ __forceinline__ uint32_t ucorrect(const uint32_t x0, const uint32_t x1, const int32_t x2, const uint32_t d0, const uint32_t d1) {
  uint32_t q=0, add, y0, y1, y2;
  
  add=x2>>31;

  // first correction
  y0=add_cc(x0, d0);
  y1=addc_cc(x1, d1);
  y2=addc_cc(x2, 0);
  add=addc(add, 0);
  q=q-add;
  
  // second correction
  y0=add_cc(y0, d0);
  y1=addc_cc(y1, d1);
  y2=addc_cc(y2, 0);
  add=addc(add, 0);
  q=q-add;
  
  // third correction
  y0=add_cc(y0, d0);
  y1=addc_cc(y1, d1);
  y2=addc_cc(y2, 0);
  add=addc(add, 0);
  q=q-add;

  // fourth correction
  y0=add_cc(y0, d0);
  y1=addc_cc(y1, d1);
  y2=addc_cc(y2, 0);
  add=addc(add, 0);
  q=q-add;

  return q;
}

/****************************************************************
 * x must be normalized, i.e., x>=0x40000000
 * computes sqrt(x)
 ****************************************************************/
__device__ __forceinline__ uint32_t usqrt(const uint32_t x) {
  float    f;
  uint32_t a;

  if((x & 0x80000000)!=0) 
    f=__uint_as_float((x>>8) + 0x4e800000);
  else
    f=__uint_as_float((x>>7) + 0x4e000000);
  
  asm volatile("sqrt.approx.f32 %0,%1;" : "=f"(f) : "f"(f));
  
  // round the approximation up
  a=__float_as_uint(f)-0x467FFF80>>8;
  
  // single correction step
  if(0>(int32_t)(x-a*a))
    a--;
  return a;
}

/****************************************************************
 * hi must be normalized, i.e., hi>=0x40000000
 * computes sqrt(x)
 ****************************************************************/
__device__ __forceinline__ uint32_t usqrt(const uint32_t lo, const uint32_t hi) {
  uint32_t   s=usqrt(hi), shifted=s<<16;  
  uint32_t   rhi, rlo;

  // compute shifted*shifted - 2^16
  rlo=madlo_cc(shifted, shifted, 0xFFFF0000);   
  rhi=madhic(shifted, shifted, 0xFFFFFFFF);
  rlo=sub_cc(lo, rlo);
  rhi=subc(hi, rhi);

  // if rhi>=0x1FFFD, we will overflow
  if(rhi>=0x1FFFD)
    s=shifted+0xFFFF;
  else {
    rhi=uleft_wrap(rlo, rhi, 15);
    s=shifted+rhi/s;
  }
  
  // first correction
  rlo=madlo(s, s, 0);
  rhi=madhi(s, s, 0);
  rlo=sub_cc(lo, rlo);
  rhi=subc(hi, rhi);
  s+=(0>(int32_t)rhi) ? 0xFFFFFFFF : 0;
  
  // second correction
  rlo=madlo(s, s, 0);
  rhi=madhi(s, s, 0);
  rlo=sub_cc(lo, rlo);
  rhi=subc(hi, rhi);
  s+=(0>(int32_t)rhi) ? 0xFFFFFFFF : 0;

  return s;
}

__device__ __forceinline__ uint32_t usqrt_div(const uint32_t lo, const uint32_t hi, const uint32_t d, const uint32_t approx) {
  uint32_t x0, x1;
  
  x0=add_cc(lo, 1);
  x1=addc(hi, 0);
  
  if(x1>=2)
    return 0xFFFFFFFF;
    
  x1=uleft_wrap(x0, x1, 31);
  x0=x0<<31;
  return udiv(x0, x1, d, approx);
}

} /* namespace cgbn */