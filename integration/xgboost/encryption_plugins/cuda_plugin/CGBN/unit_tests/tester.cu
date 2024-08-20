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
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "types.h"
#include "sizes.h"

typedef enum test_enum {
  test_set_1, test_swap_1, test_add_1, test_negate_1, test_sub_1,
  test_mul_1, test_mul_high_1, test_sqr_1, test_sqr_high_1, test_div_1, test_rem_1,
  test_div_rem_1, test_sqrt_1, test_sqrt_rem_1, test_equals_1, test_equals_2, test_equals_3, test_compare_1, test_compare_2,
  test_compare_3, test_compare_4, test_extract_bits_1, test_insert_bits_1,
  
  test_get_ui32_set_ui32_1, test_add_ui32_1, test_sub_ui32_1, test_mul_ui32_1, test_div_ui32_1, test_rem_ui32_1, 
  test_equals_ui32_1, test_equals_ui32_2, test_equals_ui32_3, test_equals_ui32_4, test_compare_ui32_1, test_compare_ui32_2,
  test_extract_bits_ui32_1, test_insert_bits_ui32_1, test_binary_inverse_ui32_1, test_gcd_ui32_1,
  
  test_mul_wide_1, test_sqr_wide_1, test_div_wide_1, test_rem_wide_1, test_div_rem_wide_1, test_sqrt_wide_1, test_sqrt_rem_wide_1,
  
  test_bitwise_and_1, test_bitwise_ior_1, test_bitwise_xor_1, test_bitwise_complement_1, test_bitwise_select_1, test_bitwise_mask_copy_1,
  test_bitwise_mask_and_1, test_bitwise_mask_ior_1, test_bitwise_mask_xor_1, test_bitwise_mask_select_1, test_shift_left_1, 
  test_shift_right_1, test_rotate_left_1, test_rotate_right_1, test_pop_count_1, test_clz_1, test_ctz_1,
  
  test_accumulator_1, test_accumulator_2, test_binary_inverse_1, test_gcd_1, test_modular_inverse_1, test_modular_power_1,
  test_bn2mont_1, test_mont2bn_1, test_mont_mul_1, test_mont_sqr_1, test_mont_reduce_wide_1, test_barrett_div_1,
  test_barrett_rem_1, test_barrett_div_rem_1, test_barrett_div_wide_1, test_barrett_rem_wide_1, test_barrett_div_rem_wide_1
} test_t;

template<test_t test, class params>
struct implementation {
  public:
  __device__ __forceinline__ static void run(typename types<params>::input_t *inputs, typename types<params>::output_t *outputs, int32_t instance) {
    printf("TEST NOT IMPLEMENTED! FIX ME!\n");
  }
};

#include "tests/tests.h"

static gmp_randstate_t  _state;
static uint32_t         _seed=0;
static uint32_t         _bits=0;
static uint32_t         _count=0;
static void            *_cpu_data=NULL;
static void            *_gpu_data=NULL;

#define $GPU(call) if((call)!=0) { printf("\nCall \"" #call "\" failed from %s, line %d\n", __FILE__, __LINE__); exit(1); }

void zero_words(uint32_t *x, uint32_t count) {
  int index;

  for(index=0;index<count;index++)
    x[index]=0;
}

void print_words(uint32_t *x, uint32_t count) {
  int index;

  for(index=count-1;index>=0;index--)
    printf("%08X", x[index]);
  printf("\n");
}

void copy_words(uint32_t *from, uint32_t *to, uint32_t count) {
  int index;

  for(index=0;index<count;index++)
    to[index]=from[index];
}

int compare_words(uint32_t *x, uint32_t *y, uint32_t count) {
  int index;

  for(index=count-1;index>=0;index--) {
    if(x[index]!=y[index]) {
      if(x[index]>y[index])
        return 1;
      else
        return -1;
    }
  }
  return 0;
}

void random_words(uint32_t *x, uint32_t count, gmp_randstate_t state) {
  int32_t index;

  for(index=0;index<count;index++)
    x[index]=gmp_urandomb_ui(state, 32);
}

void hard_random_words(uint32_t *x, uint32_t count, gmp_randstate_t state) {
  uint32_t values[6]={0x0, 0x1, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF};
  int32_t  offset, bit, bits, index;

  switch(gmp_urandomb_ui(state, 16)%3) {
    case 0:
      for(index=0;index<count;index++)
        x[index]=gmp_urandomb_ui(state, 32);
      break;
    case 1:
      for(index=0;index<count;index++)
        x[index]=values[gmp_urandomb_ui(state, 16)%6];
      break;
    case 2:
      zero_words(x, count);
      offset=0;
      while(offset<count*32) {
        bit=gmp_urandomb_ui(state, 16)%2;
        bits=gmp_urandomb_ui(state, 32)%(32*count/2)+16;
        if(bit==1) {
          if(bits>count*32-offset)
            bits=count*32-offset;
          while(bits>0) {
            if(offset%32==0 && bits>=32) {
              while(bits>=32) {
                x[offset/32]=0xFFFFFFFF;
                bits-=32;
                offset+=32;
              }
            }
            else {
              x[offset/32]=x[offset/32] + (1<<offset%32);
              bits--;
              offset++;
            }
          }
        }
        else
          offset+=bits;
      }
      break;
  }
}

template<class params>
static void generate_data(uint32_t count) {
  typename types<params>::input_t *inputs;
  int32_t                          instance;
    
  // printf("generating %d\n", params::size);
  if(_cpu_data!=NULL) {
    free(_cpu_data);
    _cpu_data=NULL;
  }
  if(_gpu_data!=NULL) {
    $GPU(cudaFree(_gpu_data));
    _gpu_data=NULL;    
  }
  _cpu_data=malloc(sizeof(typename types<params>::input_t)*count);
    
  inputs=(typename types<params>::input_t *)_cpu_data;
  gmp_randseed_ui(_state, _seed);    
  for(instance=0;instance<count;instance++) {
    hard_random_words(inputs[instance].h1._limbs, params::size/32, _state);
    hard_random_words(inputs[instance].h2._limbs, params::size/32, _state);
    random_words(inputs[instance].x1._limbs, params::size/32, _state);
    random_words(inputs[instance].x2._limbs, params::size/32, _state);
    random_words(inputs[instance].x3._limbs, params::size/32, _state);
    random_words(inputs[instance].u, 32, _state);
  }
  $GPU(cudaMalloc((void **)&_gpu_data, sizeof(typename types<params>::input_t)*count));
  $GPU(cudaMemcpy(_gpu_data, _cpu_data, sizeof(typename types<params>::input_t)*count, cudaMemcpyHostToDevice));    
}
  
template<class params>
static typename types<params>::input_t *cpu_data(uint32_t count) {
  if(params::size!=_bits || count>_count || _gpu_data==NULL) {
    if(_seed==0) {
      _seed=time(NULL);
      gmp_randinit_default(_state);
    }
    generate_data<params>(count);
    _bits=params::size;
    _count=count;
  }
  return (typename types<params>::input_t *)_cpu_data;
}
  
template<class params>
static typename types<params>::input_t *gpu_data(uint32_t count) {
  if(params::size!=_bits || count>_count || _gpu_data==NULL) {
    if(_seed==0) {
      _seed=time(NULL);
      gmp_randinit_default(_state);
    }
    generate_data<params>(count);
    _bits=params::size;
    _count=count;
  }
  return (typename types<params>::input_t *)_gpu_data;
}

template<test_t TEST, class params>
__global__ void gpu_kernel(typename types<params>::input_t *inputs, typename types<params>::output_t *outputs, uint32_t count) {
  implementation<TEST, params> impl;
  int32_t                      instance=(blockIdx.x * blockDim.x + threadIdx.x)/params::TPI;

  if(instance>=count)
    return;
  impl.run(inputs, outputs, instance);
}

template<test_t TEST, class params>
void gpu_run(typename types<params>::input_t *inputs, typename types<params>::output_t *outputs, uint32_t count) {
  uint32_t TPB=(params::TPB==0) ? 128 : params::TPB;
  uint32_t TPI=params::TPI, IPB=TPB/TPI;
  uint32_t blocks=(count+IPB+1)/IPB;
  
  gpu_kernel<TEST, params><<<blocks, TPB>>>(inputs, outputs, count);
}

template<test_t TEST, class params>
void cpu_run(typename types<params>::input_t *inputs, typename types<params>::output_t *outputs, uint32_t count) {
  implementation<TEST, params> impl;
  
  #pragma omp parallel for
  for(int index=0;index<count;index++) 
    impl.run(inputs, outputs, index);
}

template<test_t TEST, class params>
bool run_test(uint32_t count) {
  typename types<params>::input_t  *cpu_inputs, *gpu_inputs;
  typename types<params>::output_t *compare, *cpu_outputs, *gpu_outputs;
  int                               instance;
  
  if(params::size>1024)
    count=count*(1024*1024/params::size)/1024;

  cpu_inputs=cpu_data<params>(count);
  gpu_inputs=gpu_data<params>(count);
  
  compare=(typename types<params>::output_t *)malloc(sizeof(typename types<params>::output_t)*count);
  cpu_outputs=(typename types<params>::output_t *)malloc(sizeof(typename types<params>::output_t)*count);

  memset(cpu_outputs, 0, sizeof(typename types<params>::output_t)*count);
  $GPU(cudaMalloc((void **)&gpu_outputs, sizeof(typename types<params>::output_t)*count));
  $GPU(cudaMemset(gpu_outputs, 0, sizeof(typename types<params>::output_t)*count));
  
  cpu_run<TEST, params>(cpu_inputs, cpu_outputs, count);
  gpu_run<TEST, params>(gpu_inputs, gpu_outputs, count);
  $GPU(cudaMemcpy(compare, gpu_outputs, sizeof(typename types<params>::output_t)*count, cudaMemcpyDeviceToHost));
  
  for(instance=0;instance<count;instance++) {
    if(compare_words(cpu_outputs[instance].r1._limbs, compare[instance].r1._limbs, params::size/32)!=0 || 
       compare_words(cpu_outputs[instance].r2._limbs, compare[instance].r2._limbs, params::size/32)!=0) {
      printf("Test failed at index %d\n", instance);
      printf("h1: ");
      print_words(cpu_inputs[instance].h1._limbs, params::size/32);
      printf("\n");
      printf("h2: ");
      print_words(cpu_inputs[instance].h2._limbs, params::size/32);
      printf("\n");
      printf("x1: ");
      print_words(cpu_inputs[instance].x1._limbs, params::size/32);
      printf("\n");
 //     printf("x2: ");
 //     print_words(cpu_inputs[instance].x2._limbs, params::size/32);
 //     printf("\n");
 //     printf("x3: ");
 //     print_words(cpu_inputs[instance].x3._limbs, params::size/32);
 //     printf("\n");
      printf("u0: %08X   u1: %08X   u2: %08X\n\n", cpu_inputs[instance].u[0], cpu_inputs[instance].u[1], cpu_inputs[instance].u[2]);
      printf("CPU R1: ");
      print_words(cpu_outputs[instance].r1._limbs, params::size/32);
      printf("\n");
      printf("GPU R1: ");
      print_words(compare[instance].r1._limbs, params::size/32);
      printf("\n");
      printf("CPU R2: ");
      print_words(cpu_outputs[instance].r2._limbs, params::size/32);
      printf("\n");
      printf("GPU R2: ");
      print_words(compare[instance].r2._limbs, params::size/32);
      printf("\n");
      return false;
    }
  }
  
  free(compare);
  free(cpu_outputs);
  $GPU(cudaFree(gpu_outputs));
  return true;
}

#define LONG_TEST   1000000
#define MEDIUM_TEST 100000
#define SHORT_TEST  10000
#define TINY_TEST   1000
#define SINGLE_TEST 1

/*
int main() {
  run_test<test_add_1, 2048>(LONG_TEST);
  run_test<test_sub_1, 2048>(LONG_TEST);
}
*/

#include "gtest/gtest.h"
#include "unit_tests.cc"

int main(int argc, char **argv) {
  int nDevice=-1, result;
  
  cudaGetDeviceCount(&nDevice);

  if(nDevice<=0) {
    printf("Error no cuda device found.  Aborting tests\n");
    exit(EXIT_FAILURE);
  }
  
  testing::InitGoogleTest(&argc, argv);
  result=RUN_ALL_TESTS();
  if(result!=0)
    printf("Please report random seed %08X along with failure\n", _seed);
  return result;
}

