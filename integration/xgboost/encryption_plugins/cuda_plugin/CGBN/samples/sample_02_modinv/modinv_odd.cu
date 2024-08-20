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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/support.h"

/************************************************************************************************
 *  This modinv example is based on the Right-Shift Algorithm for Classical Modular Inverse
 *  from the paper "New Algorithm for Classical Modular Inverse" by Robert Lorencz.
 *
 *  This algorithm only works for an odd modulus.
 *
 *  The clean-up phase of this algorithm uses a slow bit-by-bit Montgomery reduction,
 *  which can be improved quite a bit (see modinv_fast.cu).
 ************************************************************************************************/

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define BITS 1024
#define INSTANCES 100000

// Declare the instance type
typedef struct {
  cgbn_mem_t<BITS> x;
  cgbn_mem_t<BITS> m;
  cgbn_mem_t<BITS> inverse;
} instance_t;

// support routine to generate random instances
instance_t *generate_instances(uint32_t count) {
  instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);

  for(int index=0;index<count;index++) {
    random_words(instances[index].x._limbs, BITS/32);
    random_words(instances[index].m._limbs, BITS/32);
    instances[index].m._limbs[0] |= 1;                 // guarantee modulus is odd
  }
  return instances;
}

// support routine to verify the GPU results using the CPU
void verify_results(instance_t *instances, uint32_t count) {
  mpz_t x, m, computed, correct;

  mpz_init(x);
  mpz_init(m);
  mpz_init(computed);
  mpz_init(correct);

  for(int index=0;index<count;index++) {
    to_mpz(x, instances[index].x._limbs, BITS/32);
    to_mpz(m, instances[index].m._limbs, BITS/32);
    to_mpz(computed, instances[index].inverse._limbs, BITS/32);

    if(mpz_invert(correct, x, m)==0)
      mpz_set_ui(correct, 0);
    if(mpz_cmp(correct, computed)!=0) {
      printf("gpu inverse kernel failed on instance %d\n", index);
      return;
    }
  }

  mpz_clear(x);
  mpz_clear(m);
  mpz_clear(computed);
  mpz_clear(correct);

  printf("All results match\n");
}

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

// the actual kernel
__global__ void kernel_modinv_odd(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t      bn_context(cgbn_report_monitor, report, instance);   // construct a context
  env_t          bn_env(bn_context);                                  // construct an environment for 1024-bit math
  env_t::cgbn_t  m, r, s, u, v;                                       // define m, r, s, u, v as 1024-bit bignums
  int32_t        k=0, carry, compare;

  cgbn_load(bn_env, m, &(instances[instance].m));
  cgbn_load(bn_env, v, &(instances[instance].x));

  cgbn_set(bn_env, u, m);
  cgbn_set_ui32(bn_env, r, 0);
  cgbn_set_ui32(bn_env, s, 1);

  while(true) {
    k++;
    if(cgbn_get_ui32(bn_env, u)%2==0) {
      cgbn_rotate_right(bn_env, u, u, 1);
      cgbn_add(bn_env, s, s, s);
    }
    else if(cgbn_get_ui32(bn_env, v)%2==0) {
      cgbn_rotate_right(bn_env, v, v, 1);
      cgbn_add(bn_env, r, r, r);
    }
    else {
      compare=cgbn_compare(bn_env, u, v);
      if(compare>0) {
        cgbn_add(bn_env, r, r, s);
        cgbn_sub(bn_env, u, u, v);
        cgbn_rotate_right(bn_env, u, u, 1);
        cgbn_add(bn_env, s, s, s);
      }
      else if(compare<0) {
        cgbn_add(bn_env, s, s, r);
        cgbn_sub(bn_env, v, v, u);
        cgbn_rotate_right(bn_env, v, v, 1);
        cgbn_add(bn_env, r, r, r);
      }
      else
        break;
    }
  }

  if(!cgbn_equals_ui32(bn_env, u, 1))
    cgbn_set_ui32(bn_env, r, 0);
  else {
    // last r update
    carry=cgbn_add(bn_env, r, r, r);
    if(carry==1)
      cgbn_sub(bn_env, r, r, m);

    // clean up
    if(cgbn_compare(bn_env, r, m)>0)
      cgbn_sub(bn_env, r, r, m);
    cgbn_sub(bn_env, r, m, r);

    // remove k factors of two from r, using a bit by bit Montgomery reduction
    for(int32_t i=0;i<k;i++) {
      if(cgbn_get_ui32(bn_env, r)%2==1) {
        // this can carry out, in which case we do a clever trick to
        // restore the dropped carry using a rotate (rather than a shift)
        carry=cgbn_add(bn_env, r, r, m);
        cgbn_add_ui32(bn_env, r, r, carry);
      }
      cgbn_rotate_right(bn_env, r, r, 1);
    }
  }

  cgbn_store(bn_env, &(instances[instance].inverse), r);
}

int main() {
  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;

  printf("Genereating instances ...\n");
  instances=generate_instances(INSTANCES);

  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*INSTANCES));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*INSTANCES, cudaMemcpyHostToDevice));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  printf("Running GPU kernel ...\n");
  // launch with 32 threads per instance, 128 threads (4 instances) per block
  kernel_modinv_odd<<<(INSTANCES+3)/4, 128>>>(report, gpuInstances, INSTANCES);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*INSTANCES, cudaMemcpyDeviceToHost));

  printf("Verifying the results ...\n");
  verify_results(instances, INSTANCES);

  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}
