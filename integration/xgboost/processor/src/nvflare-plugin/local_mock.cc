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
#include "local_mock.h"
#include "data_set_ids.h"

void* LocalMockProcessor::ProcessHistograms(std::size_t *size, const std::vector<double>& histograms) {
    if (debug_) {
        std::cout << "ProcessHistograms called with " << histograms.size() << " entries" << std::endl;
    }
    DamEncoder encoder(kDataSetHistogramResult, true);
    encoder.AddFloatArray(histograms);
    return encoder.Finish(*size);
}

std::vector<double> LocalMockProcessor::HandleHistograms(void *buffer, std::size_t buf_size) {
    if (debug_) {
        std::cout << "HandleHistograms called with buffer size: " << buf_size << std::endl;
    }
    auto remaining = buf_size;
    char *pointer = reinterpret_cast<char *>(buffer);

    // The buffer is concatenated by AllGather. It may contain multiple DAM buffers
    std::vector<double> result;
    while (remaining > kPrefixLen) {
        DamDecoder decoder(reinterpret_cast<uint8_t *>(pointer), remaining, true);
        if (!decoder.IsValid()) {
            std::cout << "Not DAM encoded histogram ignored at offset: "
                      << static_cast<int>((pointer - reinterpret_cast<char *>(buffer))) << std::endl;
            break;
        }
        auto size = decoder.Size();
        auto histo = decoder.DecodeFloatArray();
        if (result.empty()) {
            result = histo;
        } else {
            for (int i = 0; i < result.size(); i++) {
                result[i] += histo[i];
            }
        }

        remaining -= size;
        pointer += size;
    }

    return result;
}

Buffer LocalMockProcessor::EncryptVector(const std::vector<double>& cleartext) {
    if (debug_) {
        std::cout << "Encrypt vector size: " << cleartext.size() << std::endl;
    }

    size_t size = cleartext.size() * 8;
    auto buf = malloc(size);
    char *p = reinterpret_cast<char *>(buf);
    for (double d : cleartext) {
        memcpy(p, &d, 8);
        p += 8;
    }

    return Buffer(buf, size, true);
}

std::vector<double> LocalMockProcessor::DecryptVector(const std::vector<Buffer>& ciphertext) {
    if (debug_) {
        std::cout << "Decrypt buffer size: " << ciphertext.size() << std::endl;
    }

    std::vector<double> result;

    for (auto const &v : ciphertext) {
        size_t n = v.buf_size/8;
        auto p = reinterpret_cast<double *>(v.buffer);
        for (int i = 0; i < n; i++) {
            result.push_back(p[i]);
        }
    }

    return result;
}

std::map<int, Buffer> LocalMockProcessor::AddGHPairs(const std::map<int, std::vector<int>>& sample_ids) {
    if (debug_) {
        std::cout << "Add GH Pairs for : " << sample_ids.size() << " slots" << std::endl;
    }
    
    // Can't do this in real plugin. It needs to be broken into encrypted parts
    auto gh_pairs = DecryptVector(std::vector<Buffer>{encrypted_gh_});

    auto result = std::map<int, Buffer>();
    for (auto const &entry : sample_ids) {
        auto rows = entry.second;
        double g = 0.0;
        double h = 0.0;

        for (auto row : rows) {
            g += gh_pairs[2 * row];
            h += gh_pairs[2 * row + 1];
        }
        // In real plugin, the sum should be still in encrypted state. No need to do this step
        auto encrypted_sum = EncryptVector(std::vector<double>{g, h});
        // print_buffer(reinterpret_cast<uint8_t *>(encrypted_sum.buffer), encrypted_sum.buf_size);
        result.insert({entry.first, encrypted_sum});
    }

    return result;
}


void LocalMockProcessor::FreeEncryptedData(Buffer& ciphertext) {
    if (ciphertext.allocated) {
        free(ciphertext.buffer);
    }
}
