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
#include <omp.h>
#include <gmp.h>
#include "tests.cc"
#include "stats.cc"

typedef struct {
  mpz_t x0, x1, x2;
  mpz_t o0, o1;
  mpz_t w0, w1;
  mpz_t s0;
  mpz_t r;
} g_data_t;

#include "gmp_tests.cc"

g_data_t *g_generate_data(gmp_randstate_t state, uint32_t size, uint32_t count) {
  g_data_t *data=(g_data_t *)malloc(sizeof(g_data_t)*count);

  for(int index=0;index<count;index++) {
    mpz_init(data[index].x0);
    mpz_init(data[index].x1);
    mpz_init(data[index].x2);
    mpz_init(data[index].o0);
    mpz_init(data[index].o1);
    mpz_init(data[index].w0);
    mpz_init(data[index].w1);
    mpz_init(data[index].s0);
    mpz_init(data[index].r);
  }

  for(int index=0;index<count;index++) {
    mpz_urandomb(data[index].x0, state, size);
    mpz_urandomb(data[index].x1, state, size);
    mpz_urandomb(data[index].x2, state, size);
    mpz_urandomb(data[index].o0, state, size);
    mpz_urandomb(data[index].o1, state, size);
    mpz_urandomb(data[index].w0, state, 2*size);
    mpz_urandomb(data[index].w1, state, 2*size);
    mpz_urandomb(data[index].s0, state, 512);
    mpz_setbit(data[index].o0, 0);
    mpz_setbit(data[index].o1, 0);
  }
  return data;
}

void g_free_data(g_data_t *data, uint32_t count) {
  for(int index=0;index<count;index++) {
    mpz_clear(data[index].x0);
    mpz_clear(data[index].x1);
    mpz_clear(data[index].x2);
    mpz_clear(data[index].o0);
    mpz_clear(data[index].o1);
    mpz_clear(data[index].w0);
    mpz_clear(data[index].w1);
    mpz_clear(data[index].s0);
    mpz_clear(data[index].r);
  }
  free(data);
}

void g_run_test(stats_t *stats, test_t operation, uint32_t size, g_data_t *data, uint32_t count) {
  double time;

  stats->operation=operation;
  stats->tpi=1;
  stats->size=size;
  time=omp_get_wtime();
  if(operation==gt_add)
    stats->instances=(double)g_test_add(size, data, count);
  else if(operation==gt_sub)
    stats->instances=(double)g_test_sub(size, data, count);
  else if(operation==gt_mul)
    stats->instances=(double)g_test_mul(size, data, count);
  else if(operation==gt_div_qr)
    stats->instances=(double)g_test_div_qr(size, data, count);
  else if(operation==gt_sqrt)
    stats->instances=(double)g_test_sqrt(size, data, count);
  else if(operation==gt_powm_odd)
    stats->instances=(double)g_test_powm_odd(size, data, count);
  else if(operation==gt_mont_reduce)
    stats->instances=(double)g_test_mont_reduce(size, data, count);
  else if(operation==gt_gcd)
    stats->instances=(double)g_test_gcd(size, data, count);
  else if(operation==gt_modinv)
    stats->instances=(double)g_test_modinv(size, data, count);
  else {
    printf("g_run_test for %d not implemented\n", operation);
    exit(1);
  }
  stats->time=omp_get_wtime()-time;
  stats->throughput=stats->instances/stats->time;
  stats->next=NULL;
}

#ifndef SIZE
#define SIZE 1024
#endif

#ifndef INSTANCES
#define INSTANCES 10000
#endif

#ifndef MAX_SIZES
#define MAX_SIZES 25
#endif

int main(int argc, const char *argv[]) {
  gmp_randstate_t  state;
  g_data_t        *data;
  bool             tests[GT_LAST-GT_FIRST+1];
  int              sizes[MAX_SIZES];
  bool             all_tests=true;
  int              sizes_count=0;
  stats_t         *chain, *last, *stats;

  gmp_randinit_default(state);

  for(int index=GT_FIRST;index<=GT_LAST;index++)
    tests[index-GT_FIRST]=0;

  for(int index=0;index<MAX_SIZES;index++)
    sizes[index]=0;

  for(int index=1;index<argc;index++) {
    test_t parse;
    int    size;

    parse=gt_parse(argv[index]);
    if(parse!=test_unknown) {
      if(parse>=GT_FIRST && parse<=GT_LAST) {
        tests[parse-GT_FIRST]=true;
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

  printf("Sizes: %d\n", sizes_count);
  #pragma omp parallel
  {
    printf("Thread checking in\n");
  }
  
  if(all_tests) {
    for(int32_t testIndex=GT_FIRST;testIndex<=GT_LAST;testIndex++) {
      test_t test=static_cast<test_t>(testIndex);
    
      tests[test-GT_FIRST]=true;
    }
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
    printf("... generating data ...\n");
    data=g_generate_data(state, sizes[index], INSTANCES);
    for(int32_t testIndex=GT_FIRST;testIndex<=GT_LAST;testIndex++) {
      test_t test=static_cast<test_t>(testIndex);
    
      if(tests[test-GT_FIRST]) {
        stats=(stats_t *)malloc(sizeof(stats_s));
        printf("... %s %d ...\n", test_name(test), sizes[index]);
        g_run_test(stats, test, sizes[index], data, INSTANCES);
        if(chain==NULL) 
          chain=stats;
        else
          last->next=stats;
        last=stats;
      }
    }
    g_free_data(data, INSTANCES);
  }
  printf("Done...\n");
  FILE *report=fopen("cpu_throughput_report.csv", "w");
  if(report==NULL) {
    printf("Unable to open \"cpu_throughput_report.csv\" in the local directory for writing\n");
    exit(1);
  }
  else {
    printf("Generating \"cpu_throughput_report.csv\"");
    stats_report(report, false, chain, tests, GT_FIRST, GT_LAST, sizes, sizes_count);
    fclose(report);
  }
  printf("\n\n");
  stats_report(stdout, true, chain, tests, GT_FIRST, GT_LAST, sizes, sizes_count);
  printf("\n");
}

