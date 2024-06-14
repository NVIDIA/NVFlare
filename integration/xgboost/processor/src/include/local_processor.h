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
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "dam.h"
#include "processing/processor.h"

/*! \brief A base class for all plugins that handle encryption locally */
class LocalProcessor: public processing::Processor {
 protected:
    bool active_ = false;
    std::vector<double> gh_pairs_;
    Buffer encrypted_gh_;
    std::vector<double> histo_;
    std::vector<uint32_t> cuts_;
    std::vector<int> slots_;
    bool print_timing_ = false;
    bool debug_ = false;
    bool dam_debug_ = false;

 public:
    void Initialize(bool active, std::map<std::string, std::string> params) override;

    void Shutdown() override;

    void FreeBuffer(void *buffer) override;

    void* ProcessGHPairs(size_t *size, const std::vector<double>& pairs) override;

    void* HandleGHPairs(size_t *size, void *buffer, size_t buf_size) override;

    void InitAggregationContext(const std::vector<uint32_t> &cuts, const std::vector<int> &slots) override;

    void *ProcessAggregation(size_t *size, std::map<int, std::vector<int>> nodes) override;

    std::vector<double> HandleAggregation(void *buffer, size_t buf_size) override;

    void *ProcessHistograms(size_t *size, const std::vector<double>& histograms) override;

    std::vector<double> HandleHistograms(void *buffer, size_t buf_size) override;

    void *ProcessClearAggregation(size_t *size, std::map<int, std::vector<int>>& nodes);

    void *ProcessEncryptedAggregation(size_t *size, std::map<int, std::vector<int>>& nodes);

    // Method needs to be implemented by local plugins

    /*!
     * \brief Encrypt a vector of float-pointing numbers
     * \param cleartext A vector of numbers in cleartext
     * \return A buffer with serialized ciphertext
     */
    virtual Buffer EncryptVector(const std::vector<double>& cleartext) = 0;

    /*!
     * \brief Decrypt a serialized ciphertext into an array of numbers
     * \param ciphertext A serialzied buffer of ciphertext
     * \return An array of numbers
    */
    virtual std::vector<double> DecryptVector(const std::vector<Buffer>& ciphertext) = 0;

    /*!
     * \brief Add the G&H pairs for a series of samples
     * \param sample_ids A map of slot number and an array of sample IDs
     * \return A map of the serialized encrypted sum of G and H for each slot
     *         The input and output maps must have the same size
     */
    virtual std::map<int, Buffer> AddGHPairs(const std::map<int, std::vector<int>>& sample_ids) = 0;

    /*!
     * \brief Free encrypted data buffer
     * \param ciphertext The buffer for encrypted data
     */
    virtual void FreeEncryptedData(Buffer& ciphertext) = 0;
};
