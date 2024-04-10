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
#include "processor_nvflare.h"
#include "dam.h"

const char kPluginName[] = "nvflare";

using std::vector;
using std::cout;
using std::endl;

xgboost::common::Span<int8_t> NVFlareProcessor::ProcessGHPairs(vector<double> &pairs) {
    cout << "ProcessGHPairs called with pairs size: " << pairs.size() << endl;

    DamEncoder encoder(kDataSetHGPairs);
    encoder.AddFloatArray(pairs);
    size_t size;
    auto buffer = encoder.Finish(size);

    return xgboost::common::Span<int8_t>(reinterpret_cast<int8_t *>(buffer), size);
}

xgboost::common::Span<int8_t> NVFlareProcessor::HandleGHPairs(xgboost::common::Span<int8_t> buffer) {
    cout << "HandleGHPairs called with buffer size: " << buffer.size() << " Active: " << active_ << endl;

    return buffer;
}

xgboost::common::Span<std::int8_t> NVFlareProcessor::ProcessAggregation(
        std::vector<xgboost::bst_node_t> const &nodes_to_build, xgboost::common::RowSetCollection const &row_set) {
    cout << "ProcessAggregation called" << endl;

    int64_t data_set_id;
    if (!feature_sent_) {
        data_set_id = kDataSetAggregationWithFeatures;
        feature_sent_ = true;
    } else {
        data_set_id = kDataSetAggregation;
    }

    DamEncoder encoder(data_set_id);

    // Add cuts pointers
    vector<int64_t> cuts_vec;
    for (auto value : gidx_->Cuts().Ptrs()) {
        cuts_vec.push_back(value);
    }
    encoder.AddIntArray(cuts_vec);

    int num_samples = gh_pairs_->size()/2;
    auto cuts = gidx_->Cuts().Ptrs();
    int num_features = cuts.size() - 1;

    if (data_set_id == kDataSetAggregationWithFeatures) {
        for (std::size_t f = 0; f < num_features; f++) {
            auto slot = gidx_->GetGindex(0, f);
            if (slot >= 0) {
                features_.push_back(f);
            }
        }
        encoder.AddIntArray(features_);

        vector<int64_t> bins;
        for (int i = 0; i < num_samples; i++) {
            for (auto f : features_) {
                auto slot = gidx_->GetGindex(i, f);
                bins.push_back(slot);
            }
        }
        encoder.AddIntArray(bins);
    }

    // Add nodes to build
    vector<int64_t> node_vec;
    for (auto value : nodes_to_build) {
        node_vec.push_back(value);
    }
    encoder.AddIntArray(node_vec);

    // For each node, get the row_id/slot pair
    for (auto &node_id : nodes_to_build) {
        vector<int64_t> rows;
        auto elem = row_set[node_id];
        for (auto it = elem.begin; it != elem.end; ++it) {
            auto row_id = *it;
            rows.push_back(row_id);
        }
        encoder.AddIntArray(rows);
    }

    size_t size;
    auto buffer = encoder.Finish(size);

    return xgboost::common::Span<int8_t>(reinterpret_cast<signed char *>(buffer), size);
}

std::vector<double> NVFlareProcessor::HandleAggregation(xgboost::common::Span<std::int8_t> buffer) {
    auto remaining = buffer.size();
    char *pointer = reinterpret_cast<char *>(buffer.data());

    // The buffer is concatenated by AllGather. It may contain multiple DAM buffers
    std::vector<double> result;
    auto cuts = gidx_->Cuts().Ptrs();
    auto max_slot = cuts.back();
    auto array_size = 2 * max_slot * sizeof(double);
    double *slots = static_cast<double *>(malloc(array_size));
    while (remaining > kPrefixLen) {
        DamDecoder decoder(reinterpret_cast<uint8_t *>(pointer), remaining);
        auto size = decoder.Size();
        auto node_list = decoder.DecodeIntArray();
        for (auto node : node_list) {
            memset(slots, 0, array_size);

            // Convert per-feature histo to a flat one
            for (auto f : features_) {
                auto base = cuts[f];
                auto bins = decoder.DecodeFloatArray();
                auto n = bins.size() / 2;
                for (int i = 0; i < n; i++) {
                    auto index = base + i;
                    slots[2 * index] += bins[2 * i];
                    slots[2 * index + 1] += bins[2 * i + 1];
                }
            }
            result.insert(result.end(), slots, slots + 2 * max_slot);
        }
        remaining -= size;
        pointer += size;
    }
    free(slots);

    return result;
}

extern "C" {

xgboost::processing::Processor *LoadProcessor(char *plugin_name) {
    if (strcasecmp(plugin_name, kPluginName) != 0) {
        cout << "Unknown plugin name: " << plugin_name << endl;
        return nullptr;
    }

    return new NVFlareProcessor();
}

}