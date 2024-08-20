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
 * powm (odd modulus)
 **************************************************************************/

#define window_bits 5

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_powm_odd(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t    LOOPS=LOOP_COUNT(bits, xt_powm_odd);
  bn_t       x, power, modulus, t;
  bn_local_t window[1<<window_bits];
  int32_t    index, position;
  uint32_t   np0;

  // fixed window powm algorithm
  //   requires an odd modulus
  //   requires x<modulus

  _env.load(x, &(instances[_instance].x0));
  _env.load(power, &(instances[_instance].x1));
  _env.load(modulus, &(instances[_instance].o0));
  
  _env.bitwise_mask_and(x, x, bits-1);
  _env.bitwise_mask_ior(modulus, modulus, -1);
  _env.bitwise_mask_and(power, power, 512);

  #pragma nounroll
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.negate(t, modulus);
    _env.store(window+0, t);
    
    np0=_env.bn2mont(x, x, modulus);
    _env.store(window+1, x);
    _env.set(t, x);
    
    #pragma nounroll
    for(index=2;index<(1<<window_bits);index++) {
      _env.mont_mul(x, x, t, modulus, np0);
      _env.store(window+index, x);
    }

    position=512 - (512 % window_bits);
    index=_env.extract_bits_ui32(power, position, 512 % window_bits);
    _env.load(x, window+index);
    
    while(position>0) {
      #pragma nounroll
      for(int sqr_count=0;sqr_count<window_bits;sqr_count++) {
        _env.mont_sqr(x, x, modulus, np0);
      }
      
      position=position-window_bits;
      index=_env.extract_bits_ui32(power, position, window_bits);
      _env.load(t, window+index);
      _env.mont_mul(x, x, t, modulus, np0);
    }
    
    _env.mont2bn(x, x, modulus, np0);
   }
  _env.store(&(instances[_instance].r), x);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_powm_odd_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_powm_odd(instances);
}

#undef window_bits