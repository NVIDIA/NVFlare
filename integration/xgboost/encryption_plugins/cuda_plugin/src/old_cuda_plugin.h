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

#ifndef OLD_CUDA_PLUGIN_H
#define OLD_CUDA_PLUGIN_H

#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include "paillier.h"
#include "base_plugin.h"
#include "local_plugin.h"
#include "endec.h"

#define PRECISION 1e9

namespace nvflare {

// Define a structured header for the buffer
struct OldBufferHeader {
  bool has_key;
  size_t key_size;
  size_t rand_seed_size;
};

class OldCUDAPlugin: public LocalPlugin {
  private:
    PaillierCipher<bits>* paillier_cipher_ptr_ = nullptr;
    GHPair* encrypted_gh_pairs_ = nullptr;
    Endec* endec_ptr_ = nullptr;

  public:
    explicit OldCUDAPlugin(std::vector<std::pair<std::string_view, std::string_view>> const &args): LocalPlugin(args) {
      bool fix_seed = get_bool(args, "fix_seed");
      paillier_cipher_ptr_ = new PaillierCipher<bits>(bits/2, fix_seed, debug_);
      encrypted_gh_pairs_ = nullptr;
    }

    ~OldCUDAPlugin() {
      delete paillier_cipher_ptr_;
      if (endec_ptr_ != nullptr) {
        delete endec_ptr_;
        endec_ptr_ = nullptr;
      }
    }

    void setGHPairs() {
      if (debug_) std::cout << "setGHPairs is called" << std::endl;
      const std::uint8_t* pointer = encrypted_gh_.data();

      // Retrieve header
      OldBufferHeader header;
      std::memcpy(&header, pointer, sizeof(OldBufferHeader));
      pointer += sizeof(OldBufferHeader);

      // Get key and n (if present)
      cgbn_mem_t<bits>* key_ptr;
      if (header.has_key) {
        mpz_t n;
        mpz_init(n);
        key_ptr = (cgbn_mem_t<bits>* )malloc(header.key_size);
        if (!key_ptr) {
          std::cout << "bad alloc with key_ptr" << std::endl;
          throw std::bad_alloc();
        }
        memcpy(key_ptr, pointer, header.key_size);
        store2Gmp(n, key_ptr);
        pointer += header.key_size;

        if (header.rand_seed_size != sizeof(uint64_t)) {
          free(key_ptr);
          mpz_clear(n);
          std::cout << "rand_seed_size " << header.rand_seed_size << " is wrong " << std::endl;
          throw std::runtime_error("Invalid random seed size");
        }
        uint64_t rand_seed;
        memcpy(&rand_seed, pointer, header.rand_seed_size);
        pointer += header.rand_seed_size;

        if (!paillier_cipher_ptr_->has_pub_key) {
          paillier_cipher_ptr_->set_pub_key(n, rand_seed);
        }
        mpz_clear(n);
        free(key_ptr);
      }

      // Access payload
      std::vector<std::uint8_t> payload(pointer, pointer + (encrypted_gh_.size() - (pointer - encrypted_gh_.data())));

      ck(cudaMalloc((void **)&encrypted_gh_pairs_, payload.size()));
      cudaMemcpy(encrypted_gh_pairs_, payload.data(), payload.size(), cudaMemcpyHostToDevice);
    }

    void clearGHPairs() {
      if (debug_) std::cout << "clearGHPairs is called" << std::endl;
      if (encrypted_gh_pairs_) {
        cudaFree(encrypted_gh_pairs_);
        encrypted_gh_pairs_ = nullptr;
      }
      if (debug_) std::cout << "clearGHPairs is finished" << std::endl;
    }

    Buffer createBuffer(
      bool has_key_flag,
      cgbn_mem_t<bits>* key_ptr,
      size_t key_size,
      uint64_t rand_seed,
      size_t rand_seed_size,
      cgbn_mem_t<bits>* d_ciphers_ptr,
      size_t payload_size
    ) {
        if (debug_) std::cout << "createBuffer is called" << std::endl;
        // Calculate header size and total buffer size
        size_t header_size = sizeof(OldBufferHeader);
        size_t mem_size = header_size + key_size + rand_seed_size + payload_size;

        // Allocate buffer
        void* buffer = malloc(mem_size);
        if (!buffer) {
          std::cout << "bad alloc with buffer" << std::endl;
          throw std::bad_alloc();
        }

        // Construct header
        OldBufferHeader header;
        header.has_key = has_key_flag;
        header.key_size = key_size;
        header.rand_seed_size = rand_seed_size;

        // Copy header to buffer
        memcpy(buffer, &header, header_size);

        // Copy the key (if present)
        if (has_key_flag) {
          memcpy((char*)buffer + header_size, key_ptr, key_size);
          memcpy((char*)buffer + header_size + key_size, &rand_seed, rand_seed_size);
        }

        // Copy the payload
        cudaMemcpy((char*)buffer + header_size + key_size + rand_seed_size, d_ciphers_ptr, payload_size, cudaMemcpyDeviceToHost);

        Buffer result(buffer, mem_size, true);

        return result;
    }

    Buffer EncryptVector(const std::vector<double>& cleartext) override {
      if (debug_) std::cout << "Calling EncryptVector with count " << cleartext.size() << std::endl;
      if (endec_ptr_ != nullptr) {
        delete endec_ptr_;
      }
      endec_ptr_ = new Endec(PRECISION, debug_);

      size_t count = cleartext.size();
      int byte_length = bits / 8;
      size_t mem_size = sizeof(cgbn_mem_t<bits>) * count;
      cgbn_mem_t<bits>* h_ptr=(cgbn_mem_t<bits>* )malloc(mem_size);
      if (debug_) std::cout << "h_ptr size is " << mem_size << " indata size is " << count * byte_length << std::endl;
      for (size_t i = 0; i < count; ++i) {
        mpz_t n;
        mpz_init(n);

        endec_ptr_->encode(n, cleartext[i]);
        store2Cgbn(h_ptr + i, n);

        mpz_clear(n);
      }

      cgbn_mem_t<bits>* d_plains_ptr;
      cgbn_mem_t<bits>* d_ciphers_ptr;
      ck(cudaMalloc((void **)&d_plains_ptr, mem_size));
      ck(cudaMalloc((void **)&d_ciphers_ptr, mem_size));
      cudaMemcpy(d_plains_ptr, h_ptr, mem_size, cudaMemcpyHostToDevice);

      if (!paillier_cipher_ptr_->has_prv_key) {
#ifdef TIME
      CudaTimer cuda_timer(0);
      float gen_time=0;
      cuda_timer.start();
#endif
        if (debug_) std::cout<<"Gen KeyPair with bits: " << bits << std::endl;
        paillier_cipher_ptr_->genKeypair();
#ifdef TIME
      gen_time += cuda_timer.stop();
      std::cout<<"Gen KeyPair Time "<< gen_time <<" MS"<<std::endl;
#endif
      }

      paillier_cipher_ptr_->encrypt<TPI,TPB>(d_plains_ptr, d_ciphers_ptr, count);

      // get pub_key n
      mpz_t n;
      mpz_init(n);
      size_t key_size = sizeof(cgbn_mem_t<bits>);
      paillier_cipher_ptr_->getN(n);
      store2Cgbn(h_ptr, n);
      mpz_clear(n);

      // get rand_seed
      size_t rand_seed_size = sizeof(uint64_t);
      uint64_t rand_seed = paillier_cipher_ptr_->get_rand_seed();

      Buffer result = createBuffer(true, h_ptr, key_size, rand_seed, rand_seed_size, d_ciphers_ptr, mem_size);

      cudaFree(d_plains_ptr);
      cudaFree(d_ciphers_ptr);
      free(h_ptr);

      return result;
    }

    std::vector<double> DecryptVector(const std::vector<Buffer>& ciphertext) override {
      if (debug_) std::cout << "Calling DecryptVector" << std::endl;
      size_t mem_size = 0;
      for (int i = 0; i < ciphertext.size(); ++i) {
        mem_size += ciphertext[i].buf_size;
        if (ciphertext[i].buf_size != 2 * sizeof(cgbn_mem_t<bits>)) {
          std::cout << "buf_size is " << ciphertext[i].buf_size << std::endl;
          std::cout << "expected buf_size is " << 2 * sizeof(cgbn_mem_t<bits>) << std::endl;
          std::cout << "Fatal Error" << std::endl;
        }
      }

      size_t count = mem_size / sizeof(cgbn_mem_t<bits>);
      cgbn_mem_t<bits>* h_ptr=(cgbn_mem_t<bits>* )malloc(mem_size);
      if (debug_) std::cout << "h_ptr size is " << mem_size << " how many gh is " << count << std::endl;
      

      cgbn_mem_t<bits>* d_plains_ptr;
      cgbn_mem_t<bits>* d_ciphers_ptr;
      ck(cudaMalloc((void **)&d_plains_ptr, mem_size));
      ck(cudaMalloc((void **)&d_ciphers_ptr, mem_size));
      
      size_t offset = 0;
      for (int i = 0; i < ciphertext.size(); ++i) {
        cudaMemcpy(d_ciphers_ptr + offset, ciphertext[i].buffer, ciphertext[i].buf_size, cudaMemcpyHostToDevice);
        offset += ciphertext[i].buf_size / sizeof(cgbn_mem_t<bits>);
      }

      if (!paillier_cipher_ptr_->has_prv_key) {
        std::cout << "Can't call DecryptVector if paillier does not have private key." << std::endl;
        throw std::runtime_error("Can't call DecryptVector if paillier does not have private key.");
      }

      paillier_cipher_ptr_->decrypt<TPI,TPB>(d_ciphers_ptr, d_plains_ptr, count);

      cudaMemcpy(h_ptr, d_plains_ptr, mem_size, cudaMemcpyDeviceToHost);
      std::vector<double> result;
      result.resize(count);
      for (size_t i = 0; i < count; ++i) {
        mpz_t n;
        mpz_init(n);
        store2Gmp(n, h_ptr + i);
        double output_num = endec_ptr_->decode(n);
        result[i] = output_num;
        mpz_clear(n);
      }
      cudaFree(d_plains_ptr);
      cudaFree(d_ciphers_ptr);
      free(h_ptr);
      return result;
    }

    void AddGHPairs(std::vector<Buffer>& result, const std::uint64_t *ridx, const std::size_t size) override {
      if (debug_) std::cout << "Calling AddGHPairs with size " << size << std::endl;
      if (!encrypted_gh_pairs_) {
        setGHPairs();
      }

      std::vector<std::vector<int>> binIndexVec;
      prepareBinIndexVec(binIndexVec, ridx, size);

      GHPair* d_res_ptr;
      size_t mem_size = sizeof(GHPair);
      if (mem_size != 2 * sizeof(cgbn_mem_t<bits>)) {
        std::cout << "Fatal Error" << std::endl;
      }
      ck(cudaMalloc((void **)&d_res_ptr, mem_size));

      if (!paillier_cipher_ptr_->has_pub_key) {
        std::cout << "Can't call AddGHPairs if paillier does not have public key." << std::endl;
        throw std::runtime_error("Can't call AddGHPairs if paillier does not have public key.");
      }

      for (auto i = 0; i < binIndexVec.size(); i++) {
        const int* sample_id = binIndexVec[i].data();
        int count = binIndexVec[i].size();

        int* sample_id_d;
        ck(cudaMalloc((void **)&sample_id_d, sizeof(int) * count));
        cudaMemcpy(sample_id_d, sample_id, sizeof(int) * count, cudaMemcpyHostToDevice);

        paillier_cipher_ptr_->sum<TPI,TPB>(d_res_ptr, encrypted_gh_pairs_, sample_id_d, count);

        void* data = malloc(mem_size);
        cudaMemcpy(data, d_res_ptr, mem_size, cudaMemcpyDeviceToHost);
        Buffer buffer(data, mem_size, true);
        result[i] = buffer; // Add the Buffer object to the result map
        cudaFree(sample_id_d);
      }
      cudaFree(d_res_ptr);
      if (debug_) std::cout << "Finish AddGHPairs" << std::endl;
      if (encrypted_gh_pairs_) {
        clearGHPairs();
      }

    }
};
} // namespace nvflare

#endif // OLD_CUDA_PLUGIN_H
