/**
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
#include <cstring>
#include <thread>
#include <chrono>
#include <functional>
#include "ipcl_processor.h"

const double kScaleFactor = 1000000.0;
const bool kVerifySerialization = true;

const int kDataSetEncryptedGHPairs = 101;

std::vector<std::pair<int, int>> distribute_work(size_t num_jobs, size_t num_workers) {
    std::vector<std::pair<int, int>> result;
    auto num = num_jobs/num_workers;
    auto remainder = num_jobs%num_workers;
    int start = 0;
    for (int i = 0; i < num_workers; i++) {
        auto stop = (int)(start + num - 1);
        if (i < remainder) {
            // If jobs cannot be evenly distributed, first few workers take an extra one
            stop += 1;
        }

        if (start <= stop) {
            result.emplace_back(start, stop);
        }
        start = stop + 1;
    }

    // Verify all jobs are distributed
    int sum = 0;
    for (auto &item : result) {
        sum += item.second - item.first + 1;
    }

    if (sum != num_jobs) {
        std::cout << "Distribution error" << std::endl;
    }

    return result;
}

uint32_t to_int(double d) {
    int32_t int_val = (int32_t)(d*kScaleFactor);
    return (uint32_t)int_val;
}

double to_double(uint32_t i) {
    int32_t int_val = (int32_t)i;
    return (double)(int_val/kScaleFactor);
}

ipcl::PlainText encode_pair(double g, double h) {
    auto g_int = to_int(g);
    auto h_int = to_int(h);
    return ipcl::PlainText((std::vector<uint32_t>({g_int, h_int})));
}

std::pair<double, double> decode_pair(ipcl::PlainText &pt) {
    auto g = to_double(pt.getElementVec(0)[0]);
    auto h = to_double(pt.getElementVec(1)[0]);
    return std::pair(g, h);
}

Buffer serialize_ciphertext(ipcl::CipherText& ct) {
    std::ostringstream os;
    ipcl::serializer::serialize(os, ct);
    auto serialized_data = os.str();
    auto buf = malloc(serialized_data.size());
    memcpy(buf, serialized_data.data(), serialized_data.size());
    return Buffer(buf, serialized_data.size(), true);
}

ipcl::CipherText deserialize_ciphertext(const Buffer &buf, const ipcl::PublicKey& public_key) {
    std::string str(reinterpret_cast<char *>(buf.buffer), buf.buf_size);
    std::istringstream is(str);
    // There is a bug in IPCL to deserialize CipherText
    ipcl::PlainText pt;
    ipcl::serializer::deserialize(is, pt);
    return ipcl::CipherText(public_key, pt.getTexts());
}

void IpclProcessor::Initialize(bool active, std::map<std::string, std::string> params)  {
    active_ = active;

    ipcl::initializeContext("CPU");
    ipcl::setHybridMode(ipcl::HybridMode::OPTIMAL);

    if (active_) {
        key_ = ipcl::generateKeypair(1024, true);
        public_key_ = key_.pub_key;
        zero_ = key_.pub_key.encrypt(encode_pair(0.0, 0.0));
    }

    num_threads_ = 16;
}

void encryption_task(int first, int last, ipcl::KeyPair &key, const ipcl::CipherText& zero,
                     const std::vector<double> &gh, std::vector<Buffer> &result) {
    for (int i = first; i <= last; i++) {
        auto pt = encode_pair(gh[2 * i], gh[2 * i + 1]);
        auto ct = key.pub_key.encrypt(pt);
        result[i] = serialize_ciphertext(ct);

        if (kVerifySerialization) {
            auto new_ct = deserialize_ciphertext(result[i], key.pub_key);
            for (int j = 0; j < 2; j++) {
                std::vector<uint32_t> v1 = ct.getElementVec(j);
                std::vector<uint32_t> v2 = new_ct.getElementVec(j);
                for (int k = 0; k < v1.size(); k++) {
                    if (v1[k] != v2[k]) {
                        std::cout << "Result differs at " << i << " Size " << result[i].buf_size << std::endl;
                        break;
                    }
                }
            }
        }
    }
}

Buffer IpclProcessor::EncryptVector(const std::vector<double>& cleartext) {
    auto num = cleartext.size()/2;
    auto pairs = std::vector<Buffer>(num);

    std::cout << "Encryption starts with " << num  << " pairs" << std::endl;
    auto start = std::chrono::system_clock::now();

    auto workers = distribute_work(num, num_threads_);
    std::vector<std::thread> threads;
    for (auto& worker : workers) {
        threads.emplace_back(encryption_task, worker.first, worker.second,
                             std::ref(key_), std::ref(zero_), std::ref(cleartext), std::ref(pairs));
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Encryption time: " << duration.count() << " seconds for " << num << " GH pairs" << std::endl;

    DamEncoder encoder(kDataSetEncryptedGHPairs, true);

    auto key_bits = public_key_.getBits();
    std::vector<int64_t> key_bits_array({key_bits});
    encoder.AddIntArray(key_bits_array);

    std::ostringstream os;
    ipcl::serializer::serialize(os, public_key_);
    auto data = os.str();
    Buffer buffer(data.data(), data.size());
    encoder.AddBuffer(buffer);

    encoder.AddBufferArray(pairs);
    size_t size;
    auto buf = encoder.Finish(size);

    // Free all memory
    for (auto &item : pairs) {
        FreeEncryptedData(item);
    }
    pairs.clear();

    return Buffer(buf, size, true);
}

void decryption_task(int first, int last, ipcl::KeyPair &key,
                     const std::vector<Buffer> &ciphertext, std::vector<double> &result) {
    for (int i = first; i <= last; i++) {
        double g;
        double h;
        auto& buf = ciphertext[i];
        if (buf.buf_size == 0) {
            g = 0.0;
            h = 0.0;
        } else {
            auto ct = deserialize_ciphertext(buf, key.pub_key);
            ipcl::PlainText pt = key.priv_key.decrypt(ct);
            auto p = decode_pair(pt);
            g = p.first;
            h = p.second;
        }
        result[2*i] = g;
        result[2*i+1] = h;
    }
}

std::vector<double> IpclProcessor::DecryptVector(const std::vector<Buffer>& ciphertext) {
    std::cout << "Decrypt buffer size: " << ciphertext.size() << std::endl;

    auto num = ciphertext.size();
    auto result = std::vector<double>(2*num);

    if (!active_) {
        std::cout << "Can't decrypt on non-active node" << std::endl;
        return result;
    }

    std::cout << "Decryption starts" << std::endl;
    auto start = std::chrono::system_clock::now();
    auto workers = distribute_work(num, num_threads_);

    std::vector<std::thread> threads;
    for (auto& worker : workers) {
        threads.emplace_back(decryption_task, worker.first, worker.second, std::ref(key_),
                             std::ref(ciphertext), std::ref(result));
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Decryption time: " << duration.count() << " seconds for " << num << " GH pairs" << std::endl;

    return result;
}

void addition_task(int first, int last, const ipcl::CipherText& zero, const std::vector<ipcl::CipherText>& gh,
                   const std::map<int, std::vector<int>>& sample_ids, const std::vector<int>& keys,
                   std::map<int, Buffer>& result) {
    for (int i = first; i <= last; i++) {

        auto& samples = sample_ids.at(keys[i]);

        ipcl::CipherText sum;
        if (samples.empty()) {
            sum = zero;
        } else {
            sum = gh[samples[0]];
            for (int j=1; j < samples.size(); j++) {
                sum = sum + gh[samples[j]];
            }
        }
        result[i] = serialize_ciphertext(sum);
    }
}


std::map<int, Buffer> IpclProcessor::AddGHPairs(const std::map<int, std::vector<int>>& sample_ids) {

    std::vector<int> keys;
    keys.reserve(sample_ids.size());
    for(const auto& item : sample_ids) {
        keys.push_back(item.first);
    }

    std::cout << "AddGHPairs called with buffer size: " << encrypted_gh_.buf_size << std::endl;

    DamDecoder decoder(reinterpret_cast<uint8_t *>(encrypted_gh_.buffer), encrypted_gh_.buf_size);
    auto key_bits = decoder.DecodeIntArray()[0];
    auto key_buf = decoder.DecodeBuffer();
    // Need public key to deserialize ciphertext
    std::vector<Ipp32u> vec(key_bits / 32, 0);
    BigNumber bn(vec.data(), key_bits / 32);
    ipcl::PublicKey pub_key(bn, key_bits);
    std::string str(reinterpret_cast<char *>(key_buf.buffer), key_buf.buf_size);
    std::istringstream is(str);
    ipcl::serializer::deserialize(is, pub_key);
    public_key_ = pub_key;
    zero_ = pub_key.encrypt(encode_pair(0.0, 0.0));

    auto gh_buffers = decoder.DecodeBufferArray();
    std::vector<ipcl::CipherText> gh_ciphertext;
    gh_ciphertext.reserve(gh_buffers.size());
    std::cout << "GH array size: " << gh_buffers.size() << std::endl;
    for (auto &item : gh_buffers) {
        gh_ciphertext.push_back(deserialize_ciphertext(item, pub_key));
    }

    auto result = std::map<int, Buffer>();
    for (auto &item : sample_ids) {
        result[item.first] = Buffer();
    }

    std::cout << "Addition starts" << std::endl;
    auto start = std::chrono::system_clock::now();
    auto workers = distribute_work(sample_ids.size(), num_threads_);

    std::vector<std::thread> threads;
    for (auto& worker : workers) {
        threads.emplace_back(addition_task, worker.first, worker.second, std::ref(zero_), std::ref(gh_ciphertext),
                             std::ref(sample_ids), std::ref(keys), std::ref(result));
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Addition time: " << duration.count() << " seconds for " << sample_ids.size() << " slots" << std::endl;

    return result;
}

void IpclProcessor::FreeEncryptedData(Buffer& ciphertext) {
    if (ciphertext.allocated && ciphertext.buf_size > 0) {
        free(ciphertext.buffer);
        ciphertext.buffer = nullptr;
        ciphertext.buf_size = 0;
        ciphertext.allocated = false;
    }
}
