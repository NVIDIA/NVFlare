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
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "gpu_support.h"
#include "tests.cc"
#include "stats.cc"

__host__ __device__ int64_t LOOP_COUNT(uint32_t bits, test_t test) {
  if(test==xt_add)
    return 1000*8192/bits;
  else if(test==xt_sub)
    return 1000*8192/bits;
  else if(test==xt_accumulate)
    return 1000*8192/bits;
  else if(test==xt_mul)
    return 100*8192/bits;
  else if(test==xt_div_qr)
    return 40*8192/bits;
  else if(test==xt_sqrt)
    return 40*8192/bits;
  else if(test==xt_powm_odd)
    return 8192/bits;
  else if(test==xt_mont_reduce)
    return 100*8192/bits;
  else if(test==xt_gcd)
    return 10*8192/bits;
  else if(test==xt_modinv)
    return 10*8192/bits;
  else 
    return 0;
}

void from_mpz(uint32_t *words, uint32_t count, mpz_t value) {
  size_t written;

  if(mpz_sizeinbase(value, 2)>count*32) {
    fprintf(stderr, "from_mpz failed -- result does not fit\n");
    exit(1);
  }

  mpz_export(words, &written, -1, sizeof(uint32_t), 0, 0, value);
  while(written<count)
    words[written++]=0;
}

template<uint32_t tpi, uint32_t bits>
class xmp_tester {
  public:
  
  typedef struct {
    cgbn_mem_t<bits> x0, x1, x2;
    cgbn_mem_t<bits> o0, o1;
    cgbn_mem_t<bits> w0, w1;
    cgbn_mem_t<bits> r;
  } x_instance_t;
  
  typedef cgbn_context_t<tpi>                context_t;
  typedef cgbn_env_t<context_t, bits>        env_t;
  typedef typename env_t::cgbn_t             bn_t;
  typedef typename env_t::cgbn_local_t       bn_local_t;
  typedef typename env_t::cgbn_wide_t        bn_wide_t;
  typedef typename env_t::cgbn_accumulator_t bn_accumulator_t;

  context_t _context;
  env_t     _env;
  int32_t   _instance;
  
  __device__ __forceinline__ xmp_tester(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {
  }  

  static __host__ x_instance_t *x_generate_instances(gmp_randstate_t state, uint32_t count) {
    x_instance_t *instances=(x_instance_t *)malloc(sizeof(x_instance_t)*count);
    mpz_t         value;
    
    mpz_init(value);
    for(int index=0;index<count;index++) {
      mpz_urandomb(value, state, bits);
      from_mpz(instances[index].x0._limbs, bits/32, value);
      mpz_urandomb(value, state, bits);
      from_mpz(instances[index].x1._limbs, bits/32, value);
      mpz_urandomb(value, state, bits);
      from_mpz(instances[index].x2._limbs, bits/32, value);
      
      mpz_urandomb(value, state, bits);
      mpz_setbit(value, 0);
      from_mpz(instances[index].o0._limbs, bits/32, value);
      mpz_urandomb(value, state, bits);
      mpz_setbit(value, 0);
      from_mpz(instances[index].o1._limbs, bits/32, value);

      mpz_urandomb(value, state, 2*bits);
      from_mpz(instances[index].w0._limbs, bits*2/32, value);
      mpz_urandomb(value, state, 2*bits);
      from_mpz(instances[index].w1._limbs, bits*2/32, value);
    }
    mpz_clear(value);
    
    return instances;
  }

  __device__ __forceinline__ void x_test_add(x_instance_t *instances);
  __device__ __forceinline__ void x_test_sub(x_instance_t *instances);
  __device__ __forceinline__ void x_test_accumulate(x_instance_t *instances);
  __device__ __forceinline__ void x_test_mul(x_instance_t *instances);
  __device__ __forceinline__ void x_test_div_qr(x_instance_t *instances);
  __device__ __forceinline__ void x_test_sqrt(x_instance_t *instances);
  __device__ __forceinline__ void x_test_powm_odd(x_instance_t *instances);
  __device__ __forceinline__ void x_test_mont_reduce(x_instance_t *instances);
  __device__ __forceinline__ void x_test_gcd(x_instance_t *instances);
  __device__ __forceinline__ void x_test_modinv(x_instance_t *instances);
};

#include "xmp_tests.cu"
#include "xmp_test_powm.cu"

template<uint32_t tpi, uint32_t bits>
void x_run_test(test_t operation, typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  int threads=128, IPB=threads/tpi, blocks=(count+IPB-1)*tpi/threads;

  if(operation==xt_add) 
    x_test_add_kernel<tpi, bits><<<blocks, threads>>>(instances, count);
  else if(operation==xt_sub) 
    x_test_sub_kernel<tpi, bits><<<blocks, threads>>>(instances, count);
  else if(operation==xt_accumulate) 
    x_test_accumulate_kernel<tpi, bits><<<blocks, threads>>>(instances, count);
  else if(operation==xt_mul) 
    x_test_mul_kernel<tpi, bits><<<blocks, threads>>>(instances, count);
  else if(operation==xt_div_qr) 
    x_test_div_qr_kernel<tpi, bits><<<blocks, threads>>>(instances, count);
  else if(operation==xt_sqrt) 
    x_test_sqrt_kernel<tpi, bits><<<blocks, threads>>>(instances, count);
  else if(operation==xt_powm_odd) 
    x_test_powm_odd_kernel<tpi, bits><<<blocks, threads>>>(instances, count);
  else if(operation==xt_mont_reduce) 
    x_test_mont_reduce_kernel<tpi, bits><<<blocks, threads>>>(instances, count);
  else if(operation==xt_gcd) 
    x_test_gcd_kernel<tpi, bits><<<blocks, threads>>>(instances, count);
  else if(operation==xt_modinv) 
    x_test_modinv_kernel<tpi, bits><<<blocks, threads>>>(instances, count);
  else {
    printf("Unsupported operation -- needs to be added to x_run_test<...> in xmp_tester.cu\n");
    exit(1);
  }
}

template<uint32_t tpi, uint32_t bits>
void x_run_test(stats_t *stats, test_t operation, void *instances, uint32_t count, uint32_t repetitions) {
  typedef typename xmp_tester<tpi, bits>::x_instance_t x_instance_t;

  x_instance_t *gpuInstances;
  cudaEvent_t   start, stop;
  float         time;
  double        total=0;
  
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(x_instance_t)*count));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(x_instance_t)*count, cudaMemcpyHostToDevice));
  
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  stats->operation=operation;
  stats->tpi=tpi;
  stats->size=bits;
  printf("  ms:");
  // warm up run
  x_run_test<tpi, bits>(operation, (x_instance_t *)gpuInstances, count);
  for(int32_t run=0;run<repetitions;run++) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start, 0));
    x_run_test<tpi, bits>(operation, (x_instance_t *)gpuInstances, count);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf(" %0.3f", time);
    fflush(stdout);
    total=total+time;
  }
  printf("\n");
  total=total/1000.0;
  CUDA_CHECK(cudaFree(gpuInstances));
  stats->instances=((int64_t)count)*LOOP_COUNT(bits, operation);
  stats->time=total/(double)repetitions;
  stats->throughput=stats->instances/stats->time;
  stats->next=NULL;
}

bool x_supported_size(uint32_t size) {
  return size==128 || size==256 || size==512 || 
         size==1024 || size==2048 || size==3072 || size==4096 ||
         size==5120 || size==6144 || size==7168 || size==8192;
}

bool x_supported_tpi_size(uint32_t tpi, uint32_t size) {
  if(size==128 && tpi==4)
    return true;
  else if(size==256 && (tpi==4 || tpi==8))
    return true;
  else if(size==512 && (tpi==4 || tpi==8 || tpi==16))
    return true;
  else if(size==1024 && (tpi==8 || tpi==16 || tpi==32))
    return true;
  else if(size==2048 && (tpi==8 || tpi==16 || tpi==32))
    return true;
  else if(size==3072 && (tpi==16 || tpi==32))
    return true;
  else if(size==4096 && (tpi==16 || tpi==32))
    return true;
  else if(size==5120 && tpi==32)
    return true;
  else if(size==6144 && tpi==32)
    return true;
  else if(size==7168 && tpi==32)
    return true;
  else if(size==8192 && tpi==32)
    return true;
  return false;
}

void x_run_test(stats_t *stats, test_t operation, uint32_t tpi, uint32_t size, void *instances, uint32_t count, uint32_t repetitions) {
  if(!x_supported_tpi_size(tpi, size)) {
    printf("Unsupported tpi and size -- needs to be added to x_run_test in xmp_tester.cu\n");
    exit(1);
  }

  if(tpi==4 && size==128)
    x_run_test<4, 128>(stats, operation, instances, count, repetitions);

  else if(tpi==4 && size==256)
    x_run_test<4, 256>(stats, operation, instances, count, repetitions);
  else if(tpi==8 && size==256)
    x_run_test<8, 256>(stats, operation, instances, count, repetitions);

  else if(tpi==4 && size==512)
    x_run_test<4, 512>(stats, operation, instances, count, repetitions);
  else if(tpi==8 && size==512)
    x_run_test<8, 512>(stats, operation, instances, count, repetitions);
  else if(tpi==16 && size==512)
    x_run_test<16, 512>(stats, operation, instances, count, repetitions);
    
  else if(tpi==8 && size==1024)
    x_run_test<8, 1024>(stats, operation, instances, count, repetitions);
  else if(tpi==16 && size==1024)
    x_run_test<16, 1024>(stats, operation, instances, count, repetitions);
  else if(tpi==32 && size==1024)
    x_run_test<32, 1024>(stats, operation, instances, count, repetitions);
    
  else if(tpi==8 && size==2048)
    x_run_test<8, 2048>(stats, operation, instances, count, repetitions);
  else if(tpi==16 && size==2048)
    x_run_test<16, 2048>(stats, operation, instances, count, repetitions);
  else if(tpi==32 && size==2048)
    x_run_test<32, 2048>(stats, operation, instances, count, repetitions);
    
  else if(tpi==16 && size==3072)
    x_run_test<16, 3072>(stats, operation, instances, count, repetitions);
  else if(tpi==32 && size==3072)
    x_run_test<32, 3072>(stats, operation, instances, count, repetitions);

  else if(tpi==16 && size==4096)
    x_run_test<16, 4096>(stats, operation, instances, count, repetitions);
  else if(tpi==32 && size==4096)
    x_run_test<32, 4096>(stats, operation, instances, count, repetitions);

  else if(tpi==32 && size==5120)
    x_run_test<32, 5120>(stats, operation, instances, count, repetitions);

  else if(tpi==32 && size==6144)
    x_run_test<32, 6144>(stats, operation, instances, count, repetitions);

  else if(tpi==32 && size==7168)
    x_run_test<32, 7168>(stats, operation, instances, count, repetitions);

  else if(tpi==32 && size==8192)
    x_run_test<32, 8192>(stats, operation, instances, count, repetitions);

  else {
    printf("internal error -- tpi/size -- needs to be added to x_run_test in xmp_tester.cu\n");
    exit(1);
  }
}

void *x_generate_data(gmp_randstate_t state, uint32_t tpi, uint32_t size, uint32_t count) {
  if(size==128)
    return (void *)xmp_tester<32, 128>::x_generate_instances(state, count);
  else if(size==256)
    return (void *)xmp_tester<32, 256>::x_generate_instances(state, count);
  else if(size==512)
    return (void *)xmp_tester<32, 512>::x_generate_instances(state, count);
  else if(size==1024)
    return (void *)xmp_tester<32, 1024>::x_generate_instances(state, count);
  else if(size==2048)
    return (void *)xmp_tester<32, 2048>::x_generate_instances(state, count);
  else if(size==3072)
    return (void *)xmp_tester<32, 3072>::x_generate_instances(state, count);
  else if(size==4096)
    return (void *)xmp_tester<32, 4096>::x_generate_instances(state, count);
  else if(size==5120)
    return (void *)xmp_tester<32, 5120>::x_generate_instances(state, count);
  else if(size==6144)
    return (void *)xmp_tester<32, 6144>::x_generate_instances(state, count);
  else if(size==7168)
    return (void *)xmp_tester<32, 7168>::x_generate_instances(state, count);
  else if(size==8192)
    return (void *)xmp_tester<32, 8192>::x_generate_instances(state, count);
  else {
    printf("Unsupported size -- needs to be added to x_generate_data in xmp_tester.cu\n");
    exit(1);
  }
}

void x_free_data(void *data, uint32_t count) {
  free(data);
}

#ifndef SIZE
#define SIZE 1024
#endif

#ifndef INSTANCES
#define INSTANCES 200000
#endif

#ifndef MAX_SIZES
#define MAX_SIZES 25
#endif

#ifndef MAX_TPIS
#define MAX_TPIS 4
#endif

#ifndef RUNS
#define RUNS 5
#endif

int main(int argc, const char *argv[]) {
  gmp_randstate_t  state;
  void            *data;
  bool             tests[XT_LAST-XT_FIRST+1];
  int              sizes[MAX_SIZES];
  int              tpis[MAX_TPIS];
  bool             all_tests=true;
  int              sizes_count=0, tpis_count=0;
  stats_t         *chain, *last, *stats;

  gmp_randinit_default(state);

  for(int index=XT_FIRST;index<=XT_LAST;index++)
    tests[index-XT_FIRST]=0;

  for(int index=0;index<MAX_SIZES;index++)
    sizes[index]=0;

  for(int index=1;index<argc;index++) {
    test_t parse;
    int    size;

    parse=xt_parse(argv[index]);
    if(parse!=test_unknown) {
      if(parse>=XT_FIRST && parse<=XT_LAST) {
        tests[parse-XT_FIRST]=true;
        all_tests=false;
      }
      else {
        printf("test is only available for xmp\n");
        exit(1);
      }
    }
    else {
      size=atoi(argv[index]);
      if(size!=0)
        if(size==4 || size==8 || size==16 || size==32)
          tpis[tpis_count++]=size;
        else
          sizes[sizes_count++]=size;
      else {
        printf("invalid test/size: %s\n", argv[index]);
        exit(1);
      }
    }
  }

  for(int i=0;i<sizes_count;i++)
    for(int j=i+1;j<sizes_count;j++)
      if(sizes[i]>sizes[j]) {
        int s=sizes[i];

        sizes[i]=sizes[j];
        sizes[j]=s;
      }

  if(all_tests) {
    for(int32_t testIndex=XT_FIRST;testIndex<=XT_LAST;testIndex++) {
      test_t test=static_cast<test_t>(testIndex);
    
      tests[test-XT_FIRST]=true;
    }
  }
  
  if(tpis_count==0) {
    tpis[tpis_count++]=4;
    tpis[tpis_count++]=8;
    tpis[tpis_count++]=16;
    tpis[tpis_count++]=32;
  }

  if(sizes_count==0) {
    sizes[sizes_count++]=128;
    sizes[sizes_count++]=256;
    sizes[sizes_count++]=512;
    sizes[sizes_count++]=1024;
    sizes[sizes_count++]=2048;
    sizes[sizes_count++]=3072;
    sizes[sizes_count++]=4096;
    sizes[sizes_count++]=5120;
    sizes[sizes_count++]=6144;
    sizes[sizes_count++]=7168;
    sizes[sizes_count++]=8192;
  }
    
  chain=NULL;
  last=NULL;
  for(int index=0;index<sizes_count;index++) {
    if(!x_supported_size(sizes[index]))
      printf("... %d ... invalid test size ...\n", sizes[index]);
      
    printf("... generating data ...\n");
    data=x_generate_data(state, 32, sizes[index], INSTANCES);
    for(int tpi_index=0;tpi_index<tpis_count;tpi_index++) {
      for(int32_t testIndex=XT_FIRST;testIndex<=XT_LAST;testIndex++) {
        test_t test=static_cast<test_t>(testIndex);
    
        if(tests[test-XT_FIRST]) {
          stats=(stats_t *)malloc(sizeof(stats_s));
          if(!x_supported_tpi_size(tpis[tpi_index], sizes[index]))
            continue;
          printf("... %s %d:%d ... ", test_name(test), sizes[index], tpis[tpi_index]);
          fflush(stdout);
          x_run_test(stats, test, tpis[tpi_index], sizes[index], data, INSTANCES, RUNS);
          if(chain==NULL) 
            chain=stats;
          else
            last->next=stats;
          last=stats;
        }
      }
    }
    x_free_data(data, INSTANCES);
  }

  printf("Done...\n");
  FILE *report=fopen("gpu_throughput_report.csv", "w");
  if(report==NULL) {
    printf("Unable to open \"gpu_throughput_report.csv\" in the local directory for writing\n");
    exit(1);
  }
  else {
    printf("Generating \"gpu_throughput_report.csv\"");
    stats_report(report, false, chain, tests, XT_FIRST, XT_LAST, sizes, sizes_count);
    fclose(report);
  }
  printf("\n\n");
  stats_report(stdout, true, chain, tests, XT_FIRST, XT_LAST, sizes, sizes_count);
  printf("\n");
}



