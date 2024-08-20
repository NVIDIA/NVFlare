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
__device__ __forceinline__ uint32_t core_t<env>::get_ui32(const uint32_t a[LIMBS]) {
  uint32_t sync=sync_mask();
  
  return __shfl_sync(sync, a[0], 0, TPI);
}

template<class env> 
__device__ __forceinline__ void core_t<env>::set_ui32(uint32_t r[LIMBS], const uint32_t value) {
  uint32_t group_thread=threadIdx.x & TPI-1;

  r[0]=(group_thread==0) ? value : 0;
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=0;
}

template<class env>
__device__ __forceinline__ int32_t core_t<env>::add_ui32(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t add) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1, carry;
  int32_t  result;
  
  // FIX FIX FIX -- use custom algorithm for each TPI/PADDING 
  
  carry=mpadd32<LIMBS>(r, a, (group_thread==0) ? add : 0);
  result=fast_propagate_add(carry, r);
  clear_carry(r);
  return result;
}

template<class env>
__device__ __forceinline__ int32_t core_t<env>::sub_ui32(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t sub) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1, carry;
  int32_t  result;
  
  // FIX FIX FIX -- use custom algorithm for each TPI/PADDING 
  
  carry=mpsub32<LIMBS>(r, a, (group_thread==0) ? sub : 0);
  result=-fast_propagate_sub(carry, r);
  clear_carry(r);
  return result;
}

template<class env>
__device__ __forceinline__ uint32_t core_t<env>::mul_ui32(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t mul) {
  uint32_t carry;

  carry=mpmul32<LIMBS>(r, a, mul);
  carry=resolve_add(carry, r);
  clear_carry(r);
  return carry;
}

template<class env>
__device__ __forceinline__ uint32_t core_t<env>::div_ui32(uint32_t &r, const uint32_t a, const uint32_t d) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t x, bits, normalized, inv, approx, beta_k, t, rem, t0, t1;

  // back
  bits=uctz(d);
  normalized=d>>bits;
  inv=ubinary_inverse(normalized);
  t=__shfl_down_sync(sync, a, 1, TPI);
  t=(group_thread==TPI-1) ? 0 : t;
  x=uright_clamp(a, t, bits);
  
  // and forth  
  bits=uclz(normalized);
  normalized=normalized<<bits;
  approx=uapprox(normalized);

  beta_k=-normalized;
  
  // compute local remainder
  rem=urem(x<<bits, uleft_clamp(x, 0, bits), normalized, approx);  

  // integrate remainder from other threads
  #pragma unroll
  for(int32_t index=1;index<TPI;index=index+index) {
    t=__shfl_down_sync(sync, rem, index, TPI);
    t=(group_thread+index<TPI) ? t : 0;
    t0=madlo_cc(t, beta_k, rem);
    t1=madhic(t, beta_k, 0);
    rem=urem(t0, t1, normalized, approx);
    t0=madlo(beta_k, beta_k, 0);
    t1=madhi(beta_k, beta_k, 0);
    beta_k=urem(t0, t1, normalized, approx);
  }
  rem=rem>>bits;
  
  // Jebelean exact division
  r=(x-rem)*inv;
  
  // distribute final remainder
  return __shfl_sync(sync, a-r*d, 0, TPI);
}

template<class env>
__device__ __forceinline__ uint32_t core_t<env>::div_ui32(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t d) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t mod[LIMBS+1], x[LIMBS+1], bits, left_bits, right_bits, normalized, inv, approx, beta_k, t, rem, t0, t1;

  left_bits=uclz(d);
  normalized=d<<left_bits;
  approx=uapprox(normalized);
  
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    mod[index]=0;
  mod[LIMBS]=1;
  beta_k=mprem32<LIMBS+1>(mod, normalized, approx);

  // back
  right_bits=uctz(d);
  inv=ubinary_inverse(d>>right_bits);
  t=__shfl_down_sync(sync, a[0], 1, TPI);
  t=(group_thread==TPI-1) ? 0 : t;
  mpright<LIMBS>(x, a, right_bits, t);
  
  // and forth
  bits=left_bits+right_bits;
  mod[LIMBS]=uleft_clamp(x[LIMBS-1], 0, bits);
  mpleft<LIMBS>(mod, x, bits);
  
  // compute local remainder
  rem=mprem32<LIMBS+1>(mod, normalized, approx);

  // integrate remainder from other threads
  #pragma unroll
  for(int32_t index=1;index<TPI;index=index+index) {
    t=__shfl_down_sync(sync, rem, index, TPI);
    t=(group_thread+index<TPI) ? t : 0;
    t0=madlo_cc(t, beta_k, rem);
    t1=madhic(t, beta_k, 0);
    rem=urem(t0, t1, normalized, approx);
    t0=madlo(beta_k, beta_k, 0);
    t1=madhi(beta_k, beta_k, 0);
    beta_k=urem(t0, t1, normalized, approx);
  }
  rem=rem>>bits;

  // Jebelean exact division
  normalized=d>>right_bits;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    t=(rem>x[index]) ? 1 : 0;
    r[index]=(x[index]-rem)*inv;
    rem=madhi(r[index], normalized, t);
  }
  
  // distribute final remainder
  return __shfl_sync(sync, a[0]-r[0]*d, 0, TPI);
}

template<class env>
__device__ __forceinline__ uint32_t core_t<env>::rem_ui32(const uint32_t a, const uint32_t d) {
  uint32_t sync=sync_mask();
  uint32_t bits, normalized, approx, beta_k, t, rem, t0, t1;

  bits=uclz(d);
  normalized=d<<bits;
  approx=uapprox(normalized);

  beta_k=-normalized;

  // compute local remainder
  rem=urem(a<<bits, uleft_clamp(a, 0, bits), normalized, approx);

  // integrate remainders from other threads
  #pragma unroll
  for(int32_t index=1;index<TPI;index=index+index) {
    t=__shfl_down_sync(sync, rem, index, TPI);
    t0=madlo_cc(t, beta_k, rem);
    t1=madhic(t, beta_k, 0);
    rem=urem(t0, t1, normalized, approx);
    t0=madlo(beta_k, beta_k, 0);
    t1=madhi(beta_k, beta_k, 0);
    beta_k=urem(t0, t1, normalized, approx);
  }
  rem=rem>>bits;
  
  // distribute remainder
  return __shfl_sync(sync, rem, 0, TPI);
}

template<class env>
__device__ __forceinline__ uint32_t core_t<env>::rem_ui32(const uint32_t a[LIMBS], const uint32_t d) {
  uint32_t sync=sync_mask();
  uint32_t bits, normalized, approx, beta_k, t, rem, t0, t1;
  uint32_t x[LIMBS+1];

  bits=uclz(d);
  normalized=d<<bits;
  approx=uapprox(normalized);
  
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    x[index]=0;
  x[LIMBS]=1;
  beta_k=mprem32<LIMBS+1>(x, normalized, approx);
  
  x[LIMBS]=uleft_clamp(a[LIMBS-1], 0, bits);
  mpleft<LIMBS>(x, a, bits);
  
  // compute local remainder
  rem=mprem32<LIMBS+1>(x, normalized, approx);

  // integrate remainder from other threads
  #pragma unroll
  for(int32_t index=1;index<TPI;index=index+index) {
    t=__shfl_down_sync(sync, rem, index, TPI);
    t0=madlo_cc(t, beta_k, rem);
    t1=madhic(t, beta_k, 0);
    rem=urem(t0, t1, normalized, approx);
    t0=madlo(beta_k, beta_k, 0);
    t1=madhi(beta_k, beta_k, 0);
    beta_k=urem(t0, t1, normalized, approx);
  }
  rem=rem>>bits;
  
  // distribute remainder
  return __shfl_sync(sync, rem, 0, TPI);
}

template<class env>
__device__ __forceinline__ bool core_t<env>::equals_ui32(const uint32_t a[LIMBS], const uint32_t value) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1, lor, mask;

  lor=a[0] ^ ((group_thread==0) ? value : 0);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    lor=lor | a[index];
 
  mask=__ballot_sync(sync, lor==0);
  if(TPI<warpSize)
    mask=uright_wrap(mask, 0, threadIdx.x ^ group_thread) & TPI_ONES;
  return mask==TPI_ONES;
}

template<class env>
__device__ __forceinline__ int32_t core_t<env>::compare_ui32(const uint32_t a[LIMBS], const uint32_t value) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1, lor, mask;
  int32_t  result=1;

  // check for zero, ignoring least significant limb
  lor=(group_thread==0) ? 0 : a[0];
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    lor=lor | a[index];
 
  mask=__ballot_sync(sync, lor==0);
  if(TPI<warpSize)
    mask=uright_wrap(mask, 0, threadIdx.x ^ group_thread) & TPI_ONES;
    
  // if mask is TPI_ONES, then we numbers must match except for least significant limb
  if(mask==TPI_ONES) {
    if(a[0]<value)
      result=-1;
    else if(a[0]==value)
      result=0;
    result=__shfl_sync(sync, result, 0, TPI);
  }
  return result;
}



} /* namespace cgbn */