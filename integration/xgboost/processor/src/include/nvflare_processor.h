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
#include "processing/processor.h"

const int kDataSetHGPairs = 1;
const int kDataSetAggregation = 2;
const int kDataSetAggregationWithFeatures = 3;
const int kDataSetAggregationResult = 4;
const int kDataSetHistograms = 5;
const int kDataSetHistogramResult = 6;

class NVFlareProcessor: public processing::Processor {
 private:
    bool active_ = false;
    const std::map<std::string, std::string> *params_;
    std::vector<double> *gh_pairs_{nullptr};
    std::vector<uint32_t> cuts_;
    std::vector<int> slots_;
    bool feature_sent_ = false;
    std::vector<int64_t> features_;

 public:
    void Initialize(bool active, std::map<std::string, std::string> params) override {
        this->active_ = active;
        this->params_ = &params;
    }

    void Shutdown() override {
        this->gh_pairs_ = nullptr;
        this->cuts_.clear();
        this->slots_.clear();
    }

    void FreeBuffer(void *buffer) override {
        free(buffer);
    }

    void* ProcessGHPairs(size_t *size, const std::vector<double>& pairs) override;

    void* HandleGHPairs(size_t *size, void *buffer, size_t buf_size) override;

    void InitAggregationContext(const std::vector<uint32_t> &cuts, const std::vector<int> &slots) override {
        if (this->slots_.empty()) {
            this->cuts_ = std::vector<uint32_t>(cuts);
            this->slots_ = std::vector<int>(slots);
        } else {
            std::cout << "Multiple calls to InitAggregationContext" << std::endl;
        }
    }

    void *ProcessAggregation(size_t *size, std::map<int, std::vector<int>> nodes) override;

    std::vector<double> HandleAggregation(void *buffer, size_t buf_size) override;

    void *ProcessHistograms(size_t *size, const std::vector<double>& histograms) override;

    std::vector<double> HandleHistograms(void *buffer, size_t buf_size) override;
};