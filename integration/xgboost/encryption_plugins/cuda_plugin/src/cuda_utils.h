/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#pragma once

#include <getopt.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>
#include <math.h>
#include <string>
#include <gmp.h>
#include "cgbn.h"
#include <cstdlib> // For rand() function
#include <ctime>   // For time() function

/********** Constant Values **************/
const static unsigned int bits=2048;
const static unsigned int key_len=1024;

//const static unsigned int bits=4096;
//const static unsigned int key_len=2048;

//const static unsigned int bits=6144;
//const static unsigned int key_len=3072;

const static int TPB=512;
const static int TPI=32;
const static int window_bits=5;

/** Class **/
struct CgbnPair {
  cgbn_mem_t<bits> g;
  cgbn_mem_t<bits> h;
};

/*************Error Handling**************/
bool check(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        std::cout << "CUDA runtime API error " << cudaGetErrorString(e) << " at line " << iLine << " in file " << szFile << std::endl;
        exit(0);
        return false;
    }
    return true;
}
#define ck(call) check(call, __LINE__, __FILE__)

void cgbn_check(cgbn_error_report_t *report, const char *file=NULL, int32_t line=0) {
    // check for cgbn errors
    if(cgbn_error_report_check(report)) {
        printf("\n");
        printf("CGBN error occurred: %s\n", cgbn_error_string(report));

        if(report->_instance!=0xFFFFFFFF) {
            printf("Error reported by instance %d", report->_instance);
            if(report->_blockIdx.x!=0xFFFFFFFF || report->_threadIdx.x!=0xFFFFFFFF)
                printf(", ");
            if(report->_blockIdx.x!=0xFFFFFFFF)
                printf("blockIdx=(%d, %d, %d) ", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
            if(report->_threadIdx.x!=0xFFFFFFFF)
                printf("threadIdx=(%d, %d, %d)", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
            printf("\n");
        }
        else {
            printf("Error reported by blockIdx=(%d %d %d)", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
            printf("threadIdx=(%d %d %d)\n", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
        }
        if(file!=NULL)
            printf("file %s, line %d\n", file, line);
        exit(1);
    }
}
#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)

/*************Time Handling**************/
class CudaTimer{
    private:
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaStream_t stream;
        float time;
    public:
        CudaTimer(cudaStream_t stream){
            this->stream=stream;
        }
        void start(){
            ck(cudaEventCreate(&event_start));
            ck(cudaEventCreate(&event_stop));
            ck(cudaEventRecord(event_start, stream)); 
        }
        float stop(){
            ck(cudaEventRecord(event_stop,stream));
            ck(cudaEventSynchronize(event_stop));
            ck(cudaEventElapsedTime(&time, event_start, event_stop));
            ck(cudaEventDestroy(event_start));
            ck(cudaEventDestroy(event_stop));
            return time;
        }
        ~CudaTimer(){
        }
};

/**********GMP and CGBN functions***************/
void getPrimeOver(mpz_t rop, int bits, uint64_t &seed_start){
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, seed_start);
    seed_start++;
    mpz_t rand_num;
    mpz_init(rand_num);
    mpz_urandomb(rand_num, state, bits);
    //gmp_printf("rand_num:%Zd\n", rand_num);
    mpz_setbit(rand_num, bits-1);
    mpz_nextprime(rop, rand_num); 
    mpz_clear(rand_num);
}

template<unsigned int BITS>
void store2Cgbn(cgbn_mem_t<BITS> *address,  mpz_t z) {
    size_t words;
    if(mpz_sizeinbase(z, 2) > BITS) {
        printf("mpz_sizeinbase: %lu exceeds %d\n", mpz_sizeinbase(z, 2), BITS);
        exit(1);
    }

    mpz_export((uint32_t *)address, &words, -1, sizeof(uint32_t), 0, 0, z);
    while(words<(BITS+31)/32)
        ((uint32_t *)address)[words++]=0;
}

template<unsigned int BITS>
void store2Gmp(mpz_t z, cgbn_mem_t<BITS> *address ) {
    mpz_import(z, (BITS+31)/32, -1, sizeof(uint32_t), 0, 0, (uint32_t *)address);
}

template<unsigned int BITS>
void initArr(cgbn_mem_t<BITS> *address, int count, int default_value = 0, bool randomize = false){
    
    for(int i = 0; i < count; i++){
        int value;
        mpz_t n;
        mpz_init(n);
        if (randomize) {
            value = i; // rand();
        } else {
            value = default_value;
        }
        mpz_set_si(n, value);

        store2Cgbn(address + i, n);

        gmp_printf("input%d:%Zd\n", i, n);
        mpz_clear(n); 
    }
}
template<unsigned int BITS>
void printCgbn(cgbn_mem_t<BITS> *h_ptr, int print_count){
    for(int i = 0; i < print_count; i++){
        mpz_t n;
        mpz_init(n);
        store2Gmp(n, h_ptr + i);
        gmp_printf("printCgbn [%d]:%Zd\n",i,  n);
        mpz_clear(n); 
    }
}

template<unsigned int BITS>
void printDevCgbn(cgbn_mem_t<BITS> *d_ptr, int print_count, std::string name="cipher"){

    int mem_size=sizeof(cgbn_mem_t<BITS>)*print_count;
    cgbn_mem_t<BITS>* h_plains_ptr=(cgbn_mem_t<BITS>* )malloc(mem_size);
    cudaMemcpy(h_plains_ptr, d_ptr, mem_size, cudaMemcpyDeviceToHost); 

    for(int i = 0; i < print_count; i++){
        mpz_t n;
        mpz_init(n);
        store2Gmp(n, h_plains_ptr + i);
        gmp_printf("printDevCgbn %s[%d]:%Zd\n",name,i, n);
        mpz_clear(n); 
    }


    free(h_plains_ptr);
}

template<unsigned int BITS>
void printDevGH(CgbnPair *d_ptr, int print_count, std::string name="cipher"){

    int mem_size=sizeof(CgbnPair) * print_count;
    CgbnPair* h_plains_ptr=(CgbnPair *)malloc(mem_size);
    cudaMemcpy(h_plains_ptr, d_ptr, mem_size, cudaMemcpyDeviceToHost); 

    for(int i = 0; i < print_count; i++){
        mpz_t g, h;
        mpz_init(g);
        mpz_init(h);
        CgbnPair p = *(h_plains_ptr +i);
        store2Gmp(g, &p.g);
        store2Gmp(h, &p.h);
        gmp_printf("printDevCgbn %s[%d]:g %Zd, h %Zd\n",name, i, g, h);
        mpz_clear(g); 
        mpz_clear(h); 
    }

    free(h_plains_ptr);
}

template<unsigned int BITS>
void compArr(cgbn_mem_t<BITS> *a, cgbn_mem_t<BITS> *b,int count){
    int mem_size=sizeof(cgbn_mem_t<BITS>)*count;
    cgbn_mem_t<BITS>* ha=(cgbn_mem_t<BITS>* )malloc(mem_size);
    cudaMemcpy(ha, a, mem_size, cudaMemcpyDeviceToHost); 

    cgbn_mem_t<BITS>* hb=(cgbn_mem_t<BITS>* )malloc(mem_size);
    cudaMemcpy(hb, b, mem_size, cudaMemcpyDeviceToHost); 


    for(int i = 0; i < count; i++){
        int res=0;
        mpz_t na, nb;
        mpz_init(na);
        store2Gmp(na, ha + i);

        mpz_init(nb);
        store2Gmp(nb, hb + i);

        res=mpz_cmp(na, nb);
        if(res!=0){
            std::cout<<"res= "<<res<<" Incorrect at i= "<<i<<std::endl;;
            gmp_printf("  a=%Zd \n",na);
            gmp_printf("  b=%Zd \n",nb);
            exit(1);
        }

        mpz_clear(na);
        mpz_clear(nb);
    }
    std::cout<<"Correct!"<<std::endl;
}

bool compare_result(const std::vector<double> &a, const std::vector<double> &b, double eps=1e-6) {
    if (a.size() != b.size()) return false;
    for (auto i = 0; i < a.size(); ++i) {
        if (fabs(a[i] - b[i]) >= eps) {
            std::cout << "Fatal Error at position " << i << " " << a[i] << " " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

#endif // CUDA_UTILS_H
