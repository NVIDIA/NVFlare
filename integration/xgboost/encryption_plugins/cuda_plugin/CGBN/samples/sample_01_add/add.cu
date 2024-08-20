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
#include "../utility/cpu_support.h"
#include "../utility/cpu_simple_bn_math.h"
#include "../utility/gpu_support.h"

/************************************************************************************************
 *  This example performs component-wise addition of two arrays of 1024-bit bignums.
 *
 *  The example uses a number of utility functions and macros:
 *
 *    random_words(uint32_t *words, uint32_t count)
 *       fills words[0 .. count-1] with random data
 *
 *    add_words(uint32_t *r, uint32_t *a, uint32_t *b, uint32_t count) 
 *       sets bignums r = a+b, where r, a, and b are count words in length
 *
 *    compare_words(uint32_t *a, uint32_t *b, uint32_t count)
 *       compare bignums a and b, where a and b are count words in length.
 *       return 1 if a>b, 0 if a==b, and -1 if b>a
 *    
 *    CUDA_CHECK(call) is a macro that checks a CUDA result for an error,
 *    if an error is present, it prints out the error, call, file and line.
 *
 *    CGBN_CHECK(report) is a macro that checks if a CGBN error has occurred.
 *    if so, it prints out the error, and instance information
 *
 ************************************************************************************************/
 
// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define BITS 1024
#define INSTANCES 100000

// Declare the instance type
typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> sum;
} instance_t;

// support routine to generate random instances
instance_t *generate_instances(uint32_t count) {
  instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);

  for(int index=0;index<count;index++) {
    random_words(instances[index].a._limbs, BITS/32);
    random_words(instances[index].b._limbs, BITS/32);
  }
  return instances;
}

// support routine to verify the GPU results using the CPU
void verify_results(instance_t *instances, uint32_t count) {
  uint32_t correct[BITS/32];
  
  for(int index=0;index<count;index++) {
    add_words(correct, instances[index].a._limbs, instances[index].b._limbs, BITS/32);
    if(compare_words(correct, instances[index].sum._limbs, BITS/32)!=0) {
      printf("gpu add kernel failed on instance %d\n", index);
      return;
    }
  }
  printf("All results match\n");
}

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

// the actual kernel
__global__ void kernel_add(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;
  
  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t      bn_context(cgbn_report_monitor, report, instance);   // construct a context
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  a, b, r;                                             // define a, b, r as 1024-bit bignums

  cgbn_load(bn_env, a, &(instances[instance].a));      // load my instance's a value
  cgbn_load(bn_env, b, &(instances[instance].b));      // load my instance's b value
  cgbn_add(bn_env, r, a, b);                           // r=a+b
  cgbn_store(bn_env, &(instances[instance].sum), r);   // store r into sum
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
  kernel_add<<<(INSTANCES+3)/4, 128>>>(report, gpuInstances, INSTANCES);

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