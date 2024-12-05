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

#ifndef PAILLIER_H
#define PAILLIER_H

#pragma once

#include <random>
#include "cuda_utils.h"


/***********************Declare*************************/

template<unsigned int TPI, unsigned int BITS>
__global__ void gpu_encrypt(cgbn_error_report_t *report, cgbn_mem_t<BITS> *plains, cgbn_mem_t<BITS> * ciphers, int count); 

template<unsigned int TPI, unsigned int BITS>
__global__ void gpu_decrypt(cgbn_error_report_t *report, cgbn_mem_t<BITS> * plains, cgbn_mem_t<BITS> *ciphers, int count);

template<unsigned int TPI, unsigned int BITS>
__global__ void reduce_sum(cgbn_error_report_t *report, GHPair* result, GHPair* arr, int count, GHPair* zero);

template<unsigned int TPI, unsigned int BITS>
__global__ void reduce_sum_with_index(cgbn_error_report_t *report, GHPair* result, GHPair* arr,
int* sample_bin, int count, GHPair* zero);

template<unsigned int TPI, unsigned int BITS>
__global__ void add_two(cgbn_error_report_t *report, GHPair* result, GHPair* arr,
int* sample_bin, int count, GHPair* zero);


/***********************Class**********************/
template <unsigned int BITS>
struct PaillierPubKey{
    cgbn_mem_t<BITS> n;
    cgbn_mem_t<BITS> n_1;
    cgbn_mem_t<BITS> n_square;
    cgbn_mem_t<BITS> limit_int;
    cgbn_mem_t<BITS> rand_seed;
};

template <unsigned int BITS>
struct PaillierPrvKey{
    cgbn_mem_t<BITS> lamda;
    cgbn_mem_t<BITS> u;
};



__constant__ PaillierPrvKey<bits> c_PriKey;
__constant__ PaillierPubKey<bits> c_PubKey;


template <unsigned int BITS>
class PaillierCipher{
    private:
        mpz_t n_, p_, q_;
        uint64_t _rand_seed;
        bool fix_seed_ = false;

    public:
        int key_len;
        bool debug_ = false;
        bool has_pub_key = false;
        bool has_prv_key = false;
        PaillierPubKey<BITS> pub_key;
        PaillierPrvKey<BITS> prv_key;
        GHPair _zero;

    public:
        PaillierCipher(int key_len, bool fix_seed = false, bool debug = false){
            this->key_len=key_len;
            debug_ = debug;
            fix_seed_ = fix_seed;
            mpz_init(n_);
            mpz_init(p_);
            mpz_init(q_);

            if (debug_) std::cout<<"Construct PaillierCipher"<<std::endl;
        }
        ~PaillierCipher(){
            mpz_clear(n_);
            mpz_clear(p_);
            mpz_clear(q_);
        }

        void getN(mpz_t n) { mpz_set(n, n_); }

        uint64_t get_rand_seed() { return _rand_seed; }

        void set_pub_key(mpz_t &n, uint64_t rand) {
            if (debug_) {
                std::cout<<"PaillierCipher::set_pub_key" << std::endl;
                gmp_printf("n:%Zd\n", n);
                gmp_printf("rand:%d\n", rand);
            }
            mpz_set(n_, n);
            _rand_seed = rand;
            init_pub(n, rand);
            has_pub_key = true;
        }

        void set_keys(mpz_t &n, uint64_t rand, mpz_t &raw_p, mpz_t &raw_q){
            set_pub_key(n, rand);
            mpz_set(p_, raw_p);
            mpz_set(q_, raw_q);
            init_prv(n, raw_p, raw_q);
            has_prv_key = true;
        }

        void init_pub(mpz_t &n, uint64_t rand){
            if (debug_) std::cout<<"PaillierCipher::init_pub" << std::endl;
            mpz_t n_1, n_square, limit_int, rand_seed;
            mpz_init(n_1);
            mpz_init(n_square);
            mpz_init(limit_int);
            mpz_init(rand_seed);

            mpz_add_ui(n_1, n, 1);
            mpz_mul(n_square, n, n);
            mpz_div_ui(limit_int, n, 3);
            mpz_sub_ui(limit_int, limit_int, 1);
            mpz_sub(limit_int, n,limit_int);

            mpz_set_ui(rand_seed, rand);
            mpz_powm(rand_seed, rand_seed, n, n_square);

            store2Cgbn(&pub_key.n, n);
            store2Cgbn(&pub_key.n_1, n_1);
            store2Cgbn(&pub_key.n_square, n_square);
            store2Cgbn(&pub_key.limit_int, limit_int);
            store2Cgbn(&pub_key.rand_seed, rand_seed);
            store2Cgbn(&_zero.g, rand_seed);
            store2Cgbn(&_zero.h, rand_seed);
            ck(cudaMemcpyToSymbol(c_PubKey, &pub_key, sizeof(pub_key)));

            if (debug_) {
                gmp_printf("n_1:%Zd\n", n_1);
                gmp_printf("n:%Zd\n", n);
                gmp_printf("rand:%d \n", rand);
                gmp_printf("n_square:%Zd\n", n_square);
                gmp_printf("limit_int:%Zd\n", limit_int);
                gmp_printf("rand_seed:%Zd \n", rand_seed);

            }
            mpz_clear(n_1);
            mpz_clear(n_square);
            mpz_clear(limit_int);
            mpz_clear(rand_seed);

            if (debug_) std::cout<<"end PaillierCipher::init_pub" << std::endl;
        }

        void init_prv(mpz_t n, mpz_t raw_p, mpz_t raw_q){
            if (debug_) std::cout<<"PaillierCipher::init_prv" <<std::endl;
            mpz_t p, q;
            mpz_init(p);
            mpz_init(q);

            mpz_t lamda, u;
            mpz_init(lamda);
            mpz_init(u);

            if(mpz_cmp(raw_q, raw_p) < 0) {
                mpz_set(p, raw_q);
                mpz_set(q, raw_p);
            } else {
                mpz_set(p, raw_p);
                mpz_set(q, raw_q);
            }

            mpz_sub_ui(p, p, 1);
            mpz_sub_ui(q, q, 1);
            mpz_mul(lamda, p, q);
            store2Cgbn(&prv_key.lamda, lamda);

            mpz_invert(u,lamda,n);
            store2Cgbn(&prv_key.u, u);

            if (debug_) {
                gmp_printf("\np:%Zd\n", p);
                gmp_printf("q:%Zd\n", q);
                gmp_printf("\nlamda:%Zd\n", lamda);
                gmp_printf("u:%Zd\n", u);
            }

            mpz_clear(p);
            mpz_clear(q);

            mpz_clear(lamda);
            mpz_clear(u);
            if (debug_) std::cout<<"end PaillierCipher::init_prv " << std::endl;
        }

        void genKeypair(){
            if (debug_) std::cout<<"PaillierCipher::genKeypair" << std::endl;
            //Init mpz
            mpz_t p;
            mpz_t q;
            mpz_t n;

            mpz_init(p);
            mpz_init(q);
            mpz_init(n);


            //pick random (modulusbits/2)-bit primes p and q)
            std::random_device rd;           // Non-deterministic random seed
            std::mt19937 gen(rd());          // Mersenne Twister generator

            // Uniform integer distribution
            std::uniform_int_distribution<uint64_t> distribution(0, UINT64_MAX);

            // Generate a random number
            _rand_seed = distribution(gen);
            if (fix_seed_) _rand_seed = 12345;

            uint64_t seed_start = _rand_seed;
            int n_len = 0;
            while(n_len!=key_len){
                getPrimeOver(p, key_len/2, seed_start);
                mpz_set(q, p);
                while(mpz_cmp(p, q) == 0){
                    getPrimeOver(q, key_len/2, seed_start);
                    mpz_mul(n, p, q);
                    n_len = mpz_sizeinbase(n, 2);
                }
            }

            // Set Key
            set_keys(n, _rand_seed, p, q);


            if (debug_) {
                printf("Rand bits for n: %lu, key_len %d\n", mpz_sizeinbase(n, 2), key_len);
                std::cout<<"The size of data is:" <<sizeof(prv_key)<<" "<<sizeof(pub_key)<<std::endl;
            }

            //Set device variable
            ck(cudaMemcpyToSymbol(c_PriKey, &prv_key, sizeof(prv_key)));
            ck(cudaMemcpyToSymbol(c_PubKey, &pub_key, sizeof(pub_key)));
            ck(cudaDeviceSynchronize());
            ck(cudaGetLastError());


            //Free mpz
            mpz_clear(p);
            mpz_clear(q);
            mpz_clear(n);
            if (debug_) std::cout<<"end PaillierCipher::genKeypair" << std::endl;
        }


        void updateRandSeed(uint64_t rand){
            mpz_t n, n_square, rand_seed; 
            mpz_init(rand_seed);
            mpz_init(n);
            mpz_init(n_square);

            mpz_set_ui(rand_seed, rand);

            store2Gmp(n, &pub_key.n);
            store2Gmp(n_square, &pub_key.n_square);
            mpz_powm(rand_seed, rand_seed, n, n_square); 


            store2Cgbn(&pub_key.rand_seed, rand_seed); 
            store2Cgbn(&_zero.g, rand_seed); 
            store2Cgbn(&_zero.h, rand_seed); 
            if (debug_) gmp_printf("Updated rand_seed:%Zd \n", rand_seed);
            ck(cudaMemcpyToSymbol(c_PubKey, &pub_key, sizeof(pub_key)));
#ifdef DEBUG
            ck(cudaDeviceSynchronize());
            ck(cudaGetLastError());
#endif
            mpz_clear(rand_seed);
        }


        template<unsigned int TPI, unsigned int TPB>
            int encrypt(cgbn_mem_t<BITS>* d_plains_ptr, cgbn_mem_t<BITS>* d_ciphers_ptr, int count){
                int IPB=TPB/TPI;
                cgbn_error_report_t *report;
                ck(cgbn_error_report_alloc(&report));
                
#ifdef DEBUG
                ck(cudaDeviceSynchronize());
                ck(cudaGetLastError());
                std::cout<< "numBlocks: "<< (count+IPB-1)/IPB << ", threadsPerBlock: "<< TPB<<std::endl;
#endif
#ifdef TIME
                CudaTimer cuda_timer(0);
                cuda_timer.start();
#endif

                gpu_encrypt<TPI, BITS><<<(count+IPB-1)/IPB, TPB>>>(report, d_plains_ptr, d_ciphers_ptr, count); 

#ifdef DEBUG
                ck(cudaDeviceSynchronize());
                ck(cudaGetLastError());
                CGBN_CHECK(report);
#endif
#ifdef TIME
                float encrypt_time=cuda_timer.stop();
                std::cout<<"Encrypt Time (TPI="<<TPI<<" , TBP="<<TPB<<" ): "<<encrypt_time<<" MS"<<std::endl;
#endif
                ck(cgbn_error_report_free(report));
                return 0;
            }

        template<unsigned int TPI, unsigned int TPB>
            int decrypt(cgbn_mem_t<BITS>* d_ciphers_ptr, cgbn_mem_t<BITS>* d_plains_ptr,int count){
                int IPB=TPB/TPI;

                cgbn_error_report_t *report;
                ck(cgbn_error_report_alloc(&report));
#ifdef TIME
                CudaTimer cuda_timer(0);
                cuda_timer.start();
#endif
                gpu_decrypt<TPI, BITS><<<(count+IPB-1)/IPB, TPB>>>(report, d_plains_ptr, d_ciphers_ptr, count);
#ifdef DEBUG
                ck(cudaDeviceSynchronize());
                ck(cudaGetLastError());
                CGBN_CHECK(report);
                ck(cgbn_error_report_free(report));
#endif

#ifdef TIME
                float decrypt_time=cuda_timer.stop();
                std::cout<<"Decrypt Time (TPI="<<TPI<<" , TBP="<<TPB<<" ): "<<decrypt_time<<" MS"<<std::endl;
#endif
                return 0;
            }

        template<unsigned int TPI, unsigned int TPB>
            int sum(GHPair* d_res_ptr, GHPair* d_arr_ptr, int* sample_bin, int count) {
                int IPB = TPB / TPI;
                int maxBlocks = 2560;
                int numBlocks = min((count - 1) / IPB + 1, maxBlocks);
                int mem_size = numBlocks * sizeof(GHPair);
                if (count == 0) {
                    cudaMemcpy(d_res_ptr, &_zero, sizeof(GHPair), cudaMemcpyHostToDevice);
                    return 0;
                }

                cgbn_error_report_t *report;
                ck(cgbn_error_report_alloc(&report));
#ifdef TIME
                CudaTimer cuda_timer(0);
                cuda_timer.start();
#endif

                GHPair* d_res_ptr_2;
                ck(cudaMalloc((void **)&d_res_ptr_2, mem_size));
                GHPair* d_zero;
                ck(cudaMalloc((void **)&d_zero, sizeof(GHPair)));
                cudaMemcpy(d_zero, &_zero, sizeof(GHPair), cudaMemcpyHostToDevice);
                
                typedef cgbn_context_t<TPI>         context_t;
                typedef cgbn_env_t<context_t, BITS> env_t;
                typedef typename env_t::cgbn_t bn_t;
                int shmem_size = IPB * sizeof(GHPair);

#ifdef DEBUG
                std::cout << "before calling reduce_sum with GHPair and sample_bin" << std::endl;
                std::cout << "before calling reduce_sum count: " << count << " shm_size: " << shmem_size << " numBlocks: " << numBlocks << std::endl;
                std::cout << "before calling reduce_sum TPI: " << TPI << " TPB: " << TPB << " IPB: " << IPB << std::endl;
#endif

                reduce_sum_with_index<TPI, BITS><<<numBlocks, TPB, shmem_size>>>(report, d_res_ptr_2, d_arr_ptr, sample_bin, count, d_zero);

#ifdef DEBUG
                std::cout << "after calling reduce_sum" << std::endl;
#endif

                // final reduction
                if (numBlocks != 1) {
                    reduce_sum<TPI, BITS><<<1, TPB, shmem_size>>>(report, d_res_ptr, d_res_ptr_2, numBlocks, d_zero);
                } else {
                    cudaMemcpy(d_res_ptr, d_res_ptr_2, mem_size, cudaMemcpyDeviceToDevice);
                }

#ifdef DEBUG
                ck(cudaGetLastError());
                CGBN_CHECK(report);
#endif
                ck(cgbn_error_report_free(report));

#ifdef TIME
                float add_time=cuda_timer.stop();
                std::cout<<"Add Time (TPI="<<TPI<<" , TBP="<<TPB<<" ): "<<add_time<<" MS"<<std::endl;
#endif
                cudaFree(d_res_ptr_2);
                cudaFree(d_zero);
                return 0;
	    }

        template<unsigned int TPI, unsigned int TPB>
            int agg_tuple(GHPair* d_gh_pairs, int num_gh_pairs, unsigned int num_blocks) {
                cgbn_error_report_t *report;
                ck(cgbn_error_report_alloc(&report));

#ifdef TIME
                CudaTimer cuda_timer(0);
                cuda_timer.start();
#endif

                add_two<TPI, BITS><<<num_blocks, TPB>>>(report, d_gh_pairs, num_gh_pairs);

#ifdef TIME
                float add_time=cuda_timer.stop();
                std::cout<<"Add Time (TPI="<<TPI<<" , TBP="<<TPB<<" ): "<<add_time<<" MS"<<std::endl;
#endif

#ifdef DEBUG
                ck(cudaGetLastError());
                CGBN_CHECK(report);
#endif
                ck(cgbn_error_report_free(report));
                return 0;
	    }

        GHPair get_encrypted_zero(){
            return _zero;
        }

};


/***********************Kernels*************************/
template<unsigned int T_TPI, unsigned int T_BITS>
__global__  
void gpu_encrypt(cgbn_error_report_t *report, cgbn_mem_t<T_BITS> *plains, cgbn_mem_t<T_BITS> * ciphers, int count) {
    int tid=(blockIdx.x*blockDim.x + threadIdx.x)/T_TPI;
    if(tid>=count)
        return;

    static const uint32_t TPI=T_TPI;
    static const uint32_t BITS=T_BITS;
    typedef cgbn_context_t<TPI>         context_t;
    typedef cgbn_env_t<context_t, BITS> env_t;
    typedef typename env_t::cgbn_t bn_t;
    typedef typename env_t::cgbn_wide_t bn_w_t;

    context_t      bn_context(cgbn_report_monitor, report, tid);   // construct a context
    env_t          bn_env(bn_context);                     // construct an environment for 1024-bit math

    bn_t  t1, t2, t3;                             // define a, b, r as 1024-bit bignums
    cgbn_load(bn_env, t1, &(c_PubKey.n));//tn_
    cgbn_load(bn_env, t2, &(c_PubKey.limit_int));//t_tmp
    cgbn_load(bn_env, t3, plains+tid);//t_p

    int compare=cgbn_compare(bn_env, t3, t2); 
    if( (compare>=0) &&(cgbn_compare(bn_env, t3, t1) < 0)){
        cgbn_sub(bn_env, t2, t1, t3);
        cgbn_mul(bn_env, t2, t1, t2);
        cgbn_add_ui32(bn_env, t2, t2, 1);

        cgbn_load(bn_env, t3, &(c_PubKey.n_square));
        cgbn_rem(bn_env, t2, t2, t3);
        cgbn_modular_inverse(bn_env, t2, t2, t3);
    }else{
        cgbn_mul(bn_env, t2, t1, t3);
        cgbn_add_ui32(bn_env, t2, t2, 1);

        cgbn_load(bn_env, t3, &(c_PubKey.n_square));
        cgbn_rem(bn_env, t2, t2, t3);
    }

    cgbn_load(bn_env, t1, &(c_PubKey.rand_seed));

    bn_w_t r;
    cgbn_mul_wide(bn_env,r,t2, t1);
    cgbn_rem_wide(bn_env,t2,r,t3);
    cgbn_store(bn_env, ciphers  + tid, t2);
}

template<class env_t>
__device__ __forceinline__ void fixed_window_powm_odd(env_t _env, 
        typename env_t::cgbn_t &result, const typename env_t::cgbn_t &x, 
        const typename env_t::cgbn_t &power, const typename env_t::cgbn_t &modulus) {
    typename env_t::cgbn_t    t;
    typename env_t::cgbn_local_t window[1<<window_bits];
    int32_t    index, position, offset;
    uint32_t   np0;

    // conmpute x^power mod modulus, using the fixed window algorithm
    // requires:  x<modulus,  modulus is odd

    // compute x^0 (in Montgomery space, this is just 2^BITS - modulus)
    cgbnn_egate(_env, t, modulus);
    cgbn_store(_env, window+0, t);
    
    // convert x into Montgomery space, store into window table
    np0=cgbn_bn2mont(_env, result, x, modulus);
    cgbn_store(_env, window+1, result);
    cgbn_set(_env, t, result);
    
    // compute x^2, x^3, ... x^(2^window_bits-1), store into window table
    #pragma nounroll
    for(index=2;index<(1<<window_bits);index++) {
      cgbn_mont_mul(_env, result, result, t, modulus, np0);
      cgbn_store(_env, window+index, result);
    }

    // find leading high bit
    position=bits - cgbn_clz(_env, power);

    // break the exponent into chunks, each window_bits in length
    // load the most significant non-zero exponent chunk
    offset=position % window_bits;
    if(offset==0)
      position=position-window_bits;
    else
      position=position-offset;
    index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
    cgbn_load(_env, result, window+index);
    
    // process the remaining exponent chunks
    while(position>0) {
      // square the result window_bits times
      #pragma nounroll
      for(int sqr_count=0;sqr_count<window_bits;sqr_count++)
        cgbn_mont_sqr(_env, result, result, modulus, np0);
      
      // multiply by next exponent chunk
      position=position-window_bits;
      index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
      cgbn_load(_env, t, window+index);
      cgbn_mont_mul(_env, result, result, t, modulus, np0);
    }
    
    // we've processed the exponent now, convert back to normal space
    cgbn_mont2bn(_env, result, result, modulus, np0);
  }


template<class env_t>
__device__ __forceinline__ void sliding_window_powm_odd(env_t _env, 
        typename env_t::cgbn_t &result, const typename env_t::cgbn_t &x, 
        const typename env_t::cgbn_t &power, const typename env_t::cgbn_t &modulus) {
    typename env_t::cgbn_t     t, starts;
    int32_t      index, position, leading;
    uint32_t     mont_inv;
    typename env_t::cgbn_local_t  odd_powers[1<<window_bits-1];

    // conmpute x^power mod modulus, using Constant Length Non-Zero windows (CLNZ).
    // requires:  x<modulus,  modulus is odd
        
    // find the leading one in the power
    leading=bits-1-cgbn_clz(_env, power);
    if(leading>=0) {
      // convert x into Montgomery space, store in the odd powers table
      mont_inv=cgbn_bn2mont(_env, result, x, modulus);
      
      // compute t=x^2 mod modulus
      cgbn_mont_sqr(_env, t, result, modulus, mont_inv);
      
      // compute odd powers window table: x^1, x^3, x^5, ...
      cgbn_store(_env, odd_powers, result);
      #pragma nounroll
      for(index=1;index<(1<<window_bits-1);index++) {
        cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
        cgbn_store(_env, odd_powers+index, result);
      }
  
      // starts contains an array of bits indicating the start of a window
      cgbn_set_ui32(_env, starts, 0);
  
      // organize p as a sequence of odd window indexes
      position=0;
      while(true) {
        if(cgbn_extract_bits_ui32(_env, power, position, 1)==0)
          position++;
        else {
          cgbn_insert_bits_ui32(_env, starts, starts, position, 1, 1);
          if(position+window_bits>leading)
            break;
          position=position+window_bits;
        }
      }
  
      // load first window.  Note, since the window index must be odd, we have to
      // divide it by two before indexing the window table.  Instead, we just don't
      // load the index LSB from power
      index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
      cgbn_load(_env, result, odd_powers+index);
      position--;
      
      // Process remaining windows 
      while(position>=0) {
        cgbn_mont_sqr(_env, result, result, modulus, mont_inv);
        if(cgbn_extract_bits_ui32(_env, starts, position, 1)==1) {
          // found a window, load the index
          index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
          cgbn_load(_env, t, odd_powers+index);
          cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
        }
        position--;
      }
      
      // convert result from Montgomery space
      cgbn_mont2bn(_env, result, result, modulus, mont_inv);
    }
    else {
      // p=0, thus x^p mod modulus=1
      cgbn_set_ui32(_env, result, 1);
    }
  }


template<unsigned int TPI, unsigned int BITS>
__global__ void gpu_decrypt(cgbn_error_report_t *report, cgbn_mem_t<BITS> * plains, cgbn_mem_t<BITS> *ciphers, int count) {
    int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(tid>=count)
        return;

    cgbn_context_t<TPI>  bn_context(cgbn_report_monitor, report, tid);  
    cgbn_env_t<cgbn_context_t<TPI>, BITS>  bn_env(bn_context);

    typename cgbn_env_t<cgbn_context_t<TPI>, BITS>::cgbn_t t, p;
    typename cgbn_env_t<cgbn_context_t<TPI>, BITS>::cgbn_t n;

    cgbn_load(bn_env, t, ciphers + tid);
    cgbn_load(bn_env, p, &(c_PriKey.lamda));
    cgbn_load(bn_env, n, &(c_PubKey.n_square));

    //cgbn_modular_power(bn_env, t, t,p, n);
    //fixed_window_powm_odd(bn_env,t, t, p, n);
    sliding_window_powm_odd(bn_env,t, t, p, n);


    cgbn_load(bn_env, n, &(c_PubKey.n));
    cgbn_sub_ui32(bn_env, t, t, 1);

    cgbn_load(bn_env, p, &(c_PriKey.u));

    cgbn_div(bn_env, t, t, n);
    cgbn_mul(bn_env, t, t, p);
    cgbn_rem(bn_env, t, t, n);

    cgbn_store(bn_env, plains + tid, t);
}


template <unsigned int TPI, unsigned int BITS>
__global__ void reduce_sum(cgbn_error_report_t* report, GHPair* result, GHPair* arr, int count, GHPair* zero) {
    typedef cgbn_context_t<TPI> context_t;
    typedef cgbn_env_t<context_t, BITS> env_t;
    typedef typename env_t::cgbn_t bn_t;
    typedef typename env_t::cgbn_wide_t bn_w_t;

    int id = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    int shm_id = threadIdx.x / TPI;
    int IPB = blockDim.x / TPI;

    context_t bn_context(cgbn_report_monitor, report, id);
    env_t bn_env(bn_context);

    extern __shared__ GHPair sdata3[];
    bn_t a, b, c, tmp_g, tmp_h;
    bn_t n_square;
    bn_w_t r;

    int total_windows = (count - 1) / (IPB * gridDim.x) + 1;
    cgbn_load(bn_env, n_square, &c_PubKey.n_square);
    for (unsigned int window = 0; window < total_windows; window++) {
        int global_position = id + window * IPB * gridDim.x;
        if (global_position >= count) {
            // Load rand_seed into sdata3 directly for positions exceeding count
            sdata3[shm_id] = zero[0];
        } else {
            // Load pairs of elements from arr into sdata3
            sdata3[shm_id] = arr[global_position];
        }
        __syncthreads();

        // Perform reduction in shared memory
        for (unsigned int s = IPB / 2; s > 0; s >>= 1) {
            if (shm_id < s) {
                // Load pairs of elements from shared memory and perform reduction
                cgbn_load(bn_env, a, &(sdata3[shm_id].g));
                cgbn_load(bn_env, b, &(sdata3[shm_id + s].g));
                cgbn_mul_wide(bn_env, r, a, b);
                cgbn_rem_wide(bn_env, c, r, n_square);
                cgbn_store(bn_env, &(sdata3[shm_id].g), c);

                cgbn_load(bn_env, a, &(sdata3[shm_id].h));
                cgbn_load(bn_env, b, &(sdata3[shm_id + s].h));
                cgbn_mul_wide(bn_env, r, a, b);
                cgbn_rem_wide(bn_env, c, r, n_square);
                cgbn_store(bn_env, &(sdata3[shm_id].h), c);
            }
            __syncthreads();
        }

        if (shm_id == 0) {
            if (window == 0) {
                // Store the result of the first window into tmp
                cgbn_load(bn_env, tmp_g, &(sdata3[0].g));
                cgbn_load(bn_env, tmp_h, &(sdata3[0].h));
            } else {
                // Add the result of subsequent windows to tmp
                cgbn_load(bn_env, a, &(sdata3[0].g));
                cgbn_mul_wide(bn_env, r, a, tmp_g);
                cgbn_rem_wide(bn_env, tmp_g, r, n_square);

                cgbn_load(bn_env, a, &(sdata3[0].h));
                cgbn_mul_wide(bn_env, r, a, tmp_h);
                cgbn_rem_wide(bn_env, tmp_h, r, n_square);
            }
        }
        __syncthreads();
    }

    __syncthreads();

    // Write the final result for this block to global memory
    if (shm_id == 0) {
        cgbn_store(bn_env, &(result[blockIdx.x].g), tmp_g);
        cgbn_store(bn_env, &(result[blockIdx.x].h), tmp_h);
    }
    __syncthreads();
}


template <unsigned int TPI, unsigned int BITS>
__global__ void reduce_sum_with_index(cgbn_error_report_t* report, GHPair* result, GHPair* arr, int* sample_bin, int count, GHPair* zero) {
                            
    typedef cgbn_context_t<TPI> context_t;
    typedef cgbn_env_t<context_t, BITS> env_t;
    typedef typename env_t::cgbn_t bn_t;
    typedef typename env_t::cgbn_wide_t bn_w_t;

    int id = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    int shm_id = threadIdx.x / TPI;
    int IPB = blockDim.x / TPI;

    context_t bn_context(cgbn_report_monitor, report, id);
    env_t bn_env(bn_context);

    extern __shared__ GHPair sdata4[];
    bn_t a, b, c, tmp_g, tmp_h;
    bn_t n_square;
    bn_w_t r;

    int total_windows = (count - 1) / (IPB * gridDim.x) + 1;
    cgbn_load(bn_env, n_square, &c_PubKey.n_square);
    for (unsigned int window = 0; window < total_windows; window++) {
        int global_position = id + window * IPB * gridDim.x;

#ifdef DEBUG
        printf("id %d shm_id %d IPB %d threadIdx.x %d blockIdx.x %d gridDim.x %d window %d total_windows %d global_position %d \n", id, shm_id, IPB, threadIdx.x, blockIdx.x, gridDim.x, window, total_windows, global_position);
#endif
      
        if (global_position >= count) {
            // Load rand_seed into sdata4 directly for positions exceeding count
            sdata4[shm_id] = zero[0];
        } else {
            int sample_id = sample_bin[global_position];
            //printf("loading global position %d sample id %d", global_position, sample_id);
            // each shm_id copy one instance from global to shared mem
            sdata4[shm_id] = arr[sample_id];
        }
        __syncthreads();

        // Perform reduction in shared memory
        for (unsigned int s = IPB / 2; s > 0; s >>= 1) {
            if (shm_id < s) {
                // Load pairs of elements from shared memory and perform reduction
                cgbn_load(bn_env, a, &(sdata4[shm_id].g));
                cgbn_load(bn_env, b, &(sdata4[shm_id + s].g));
                cgbn_mul_wide(bn_env, r, a, b);
                cgbn_rem_wide(bn_env, c, r, n_square);
                cgbn_store(bn_env, &(sdata4[shm_id].g), c);

                cgbn_load(bn_env, a, &(sdata4[shm_id].h));
                cgbn_load(bn_env, b, &(sdata4[shm_id + s].h));
                cgbn_mul_wide(bn_env, r, a, b);
                cgbn_rem_wide(bn_env, c, r, n_square);
                cgbn_store(bn_env, &(sdata4[shm_id].h), c);
            }
            __syncthreads();
        }

        if (shm_id == 0) {
            if (window == 0) {
                // Store the result of the first window into tmp
                cgbn_load(bn_env, tmp_g, &(sdata4[0].g));
                cgbn_load(bn_env, tmp_h, &(sdata4[0].h));
            } else {
                // Add the result of subsequent windows to tmp
                cgbn_load(bn_env, a, &(sdata4[0].g));
                cgbn_mul_wide(bn_env, r, a, tmp_g);
                cgbn_rem_wide(bn_env, tmp_g, r, n_square);

                cgbn_load(bn_env, a, &(sdata4[0].h));
                cgbn_mul_wide(bn_env, r, a, tmp_h);
                cgbn_rem_wide(bn_env, tmp_h, r, n_square);
            }
        }
        __syncthreads();
    }

    __syncthreads();

    // Write the final result for this block to global memory
    if (shm_id == 0) {
        cgbn_store(bn_env, &(result[blockIdx.x].g), tmp_g);
        cgbn_store(bn_env, &(result[blockIdx.x].h), tmp_h);
    }
    __syncthreads();
}


template <unsigned int TPI, unsigned int BITS>
__global__ void add_two(cgbn_error_report_t *report, GHPair* arr, int count) {

    int id = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    int item_id = id * 2; // each tuple contains 2 GHPairs (G0, H0) and (G1, H1)

    if (item_id < count) {
        typedef cgbn_context_t<TPI> context_t;
        typedef cgbn_env_t<context_t, BITS> env_t;
        typedef typename env_t::cgbn_t bn_t;
        typedef typename env_t::cgbn_wide_t bn_w_t;

        context_t bn_context(cgbn_report_monitor, report, item_id);
        env_t bn_env(bn_context);

        bn_t a, b, c;
        bn_t n_square;
        bn_w_t r;

        cgbn_load(bn_env, n_square, &c_PubKey.n_square);

        // Add 2 GHPairs up
        // Load pairs of elements and perform reduction
        cgbn_load(bn_env, a, &(arr[item_id].g));
        cgbn_load(bn_env, b, &(arr[item_id + 1].g));
        cgbn_mul_wide(bn_env, r, a, b);
        cgbn_rem_wide(bn_env, c, r, n_square);
        cgbn_store(bn_env, &(arr[item_id].g), c);

        cgbn_load(bn_env, a, &(arr[item_id].h));
        cgbn_load(bn_env, b, &(arr[item_id + 1].h));
        cgbn_mul_wide(bn_env, r, a, b);
        cgbn_rem_wide(bn_env, c, r, n_square);
        cgbn_store(bn_env, &(arr[item_id].h), c);

    }
}

#endif // PAILLIER_H
