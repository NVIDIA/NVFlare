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


/**************************************************************************
 * Addition
 **************************************************************************/
template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_add(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t LOOPS=LOOP_COUNT(bits, xt_add);
  bn_t    x0, x1, r;
    
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));

  _env.set(r, x0);
  #pragma unroll 10
  for(int32_t loop=0;loop<LOOPS;loop++)
    _env.add(r, r, x1);

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_add_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_add(instances);
}


/**************************************************************************
 * Subtraction
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_sub(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t LOOPS=LOOP_COUNT(bits, xt_sub);
  bn_t    x0, x1, r;
    
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));

  _env.set(r, x0);
  #pragma unroll 10
  for(int32_t loop=0;loop<LOOPS;loop++)
    _env.sub(r, r, x1);

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_sub_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_sub(instances);
}


/**************************************************************************
 * Accumulate
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_accumulate(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t          LOOPS=LOOP_COUNT(bits, xt_accumulate);
  bn_t             x0, x1, r;
  bn_accumulator_t acc;
  
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));

  _env.set(acc, x0);
  #pragma nounroll
  for(int32_t loop=0;loop<LOOPS;loop++)
    _env.add(acc, x1);
  _env.resolve(r, acc);

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_accumulate_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_accumulate(instances);
}


/**************************************************************************
 * Multiplication
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_mul(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_mul);
  bn_t      x0, x1, r;
  bn_wide_t w;

  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));

  _env.set(r, x0);
  #pragma unroll 4
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.mul_wide(w, r, x0);
    _env.set(r, w._low);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_mul_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_mul(instances);
}


/**************************************************************************
 * Division
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_div_qr(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_div_qr);
  bn_t      x0, x1, q, r;
  bn_wide_t w;
  
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));

  _env.shift_right(w._high, x1, 1);
  _env.set(w._low, x0);
  
  #pragma nounroll
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.div_rem_wide(q, r, w, x1);
    _env.set(w._high, r);
    _env.set(w._low, q);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_div_qr_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_div_qr(instances);
}

/**************************************************************************
 * Square root
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_sqrt(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_sqrt);
  bn_t      r;
  bn_wide_t w;
  
  _env.load(w._low, &(instances[_instance].x0));
  _env.load(w._high, &(instances[_instance].x1));

  #pragma nounroll
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.sqrt_wide(r, w);
    _env.set(w._low, r);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_sqrt_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_sqrt(instances);
}


/**************************************************************************
 * Mont reduce
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_mont_reduce(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_mont_reduce);
  bn_t      x0, x1, o0, r;
  bn_wide_t w;
  uint32_t  mp0;
  
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));
  _env.load(o0, &(instances[_instance].o0));
  
  _env.set(w._low, x0);
  _env.set(w._high, x1);

  mp0=-_env.binary_inverse_ui32(_env.get_ui32(o0));
  
  #pragma nounrull
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.mont_reduce_wide(r, w, o0, mp0);
    _env.set(w._low, r);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_mont_reduce_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_mont_reduce(instances);
}


/**************************************************************************
 * GCD
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_gcd(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_gcd);
  bn_t      x0, x1, r;
  
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));
  
  #pragma nounroll
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.gcd(r, x0, x1);
    _env.add_ui32(x0, x0, 1);
    _env.add_ui32(x1, x1, 1);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_gcd_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_gcd(instances);
}


/**************************************************************************
 * Mod Inv
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_modinv(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_modinv);
  bn_t      x0, x1, r;
  
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));
  
  #pragma nounroll
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.modular_inverse(r, x0, x1);
    _env.add_ui32(x0, x0, 1);
    _env.add_ui32(x1, x1, 1);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_modinv_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_modinv(instances);
}
