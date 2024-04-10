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
#include <vector>
#include <map>
#include "processing/processor.h"

const int kDataSetHGPairs = 1;
const int kDataSetAggregation = 2;
const int kDataSetAggregationWithFeatures = 3;
const int kDataSetAggregationResult = 4;

class NVFlareProcessor: public xgboost::processing::Processor {
 private:
    bool active_ = false;
    const std::map<std::string, std::string> *params_;
    std::vector<double> *gh_pairs_{nullptr};
    const xgboost::GHistIndexMatrix *gidx_;
    bool feature_sent_ = false;
    std::vector<int64_t> features_;

 public:
    void Initialize(bool active, std::map<std::string, std::string> params) override {
        this->active_ = active;
        this->params_ = &params;
    }

    void Shutdown() override {
        this->gh_pairs_ = nullptr;
        this->gidx_ = nullptr;
    }

    void FreeBuffer(xgboost::common::Span<std::int8_t> buffer) override {
        free(buffer.data());
    }

    xgboost::common::Span<int8_t> ProcessGHPairs(std::vector<double> &pairs) override;

    xgboost::common::Span<int8_t> HandleGHPairs(xgboost::common::Span<int8_t> buffer) override;

    void InitAggregationContext(xgboost::GHistIndexMatrix const &gidx) override {
        this->gidx_ = &gidx;
    }

    xgboost::common::Span<std::int8_t> ProcessAggregation(std::vector<xgboost::bst_node_t> const &nodes_to_build,
                                                          xgboost::common::RowSetCollection const &row_set) override;

    std::vector<double> HandleAggregation(xgboost::common::Span<std::int8_t> buffer) override;
};