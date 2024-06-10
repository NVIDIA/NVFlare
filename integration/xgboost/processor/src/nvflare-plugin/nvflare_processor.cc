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
#include "nvflare_processor.h"
#include "dam.h"

const char kPluginName[] = "nvflare";

using std::vector;
using std::cout;
using std::endl;

void* NVFlareProcessor::ProcessGHPairs(size_t *size, const std::vector<double>& pairs) {
    cout << "ProcessGHPairs called with pairs size: " << pairs.size() << endl;
    gh_pairs_ = new std::vector<double>(pairs);

    DamEncoder encoder(kDataSetHGPairs);
    encoder.AddFloatArray(pairs);
    auto buffer = encoder.Finish(*size);

    return buffer;
}

void* NVFlareProcessor::HandleGHPairs(size_t *size, void *buffer, size_t buf_size)  {
    cout << "HandleGHPairs called with buffer size: " << buf_size << " Active: " << active_ << endl;
    *size = buf_size;
    return buffer;
}

void *NVFlareProcessor::ProcessAggregation(size_t *size, std::map<int, std::vector<int>> nodes) {
    cout << "ProcessAggregation called with " << nodes.size() << " nodes" << endl;

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
    for (auto value : cuts_) {
        cuts_vec.push_back(value);
    }
    encoder.AddIntArray(cuts_vec);

    auto num_features = cuts_.size() - 1;
    auto num_samples = slots_.size() / num_features;
    cout << "Samples: " << num_samples << " Features: " << num_features << endl;

    if (data_set_id == kDataSetAggregationWithFeatures) {
        if (features_.empty()) {
            for (std::size_t f = 0; f < num_features; f++) {
                auto slot = slots_[f];
                if (slot >= 0) {
                    features_.push_back(f);
                }
            }
        }
        cout << "Including feature size: " << features_.size() << endl;
        encoder.AddIntArray(features_);

        vector<int64_t> bins;
        for (int i = 0; i < num_samples; i++) {
            for (auto f : features_) {
                auto index = f + i * num_features;
                if (index > slots_.size()) {
                    cout << "Index is out of range " << index << endl;
                }
                auto slot = slots_[index];
                bins.push_back(slot);
            }
        }
        encoder.AddIntArray(bins);
    }

    // Add nodes to build
    vector<int64_t> node_vec;
    for (const auto &kv : nodes) {
        std::cout << "Node: " << kv.first << " Rows: " << kv.second.size() << std::endl;
        node_vec.push_back(kv.first);
    }
    encoder.AddIntArray(node_vec);

    // For each node, get the row_id/slot pair
    for (const auto &kv : nodes) {
        vector<int64_t> rows;
        for (auto row : kv.second) {
            rows.push_back(row);
        }
        encoder.AddIntArray(rows);
    }

    auto buffer = encoder.Finish(*size);
    return buffer;
}

std::vector<double> NVFlareProcessor::HandleAggregation(void *buffer, size_t buf_size) {
    cout << "HandleAggregation called with buffer size: " << buf_size << endl;
    auto remaining = buf_size;
    char *pointer = reinterpret_cast<char *>(buffer);

    // The buffer is concatenated by AllGather. It may contain multiple DAM buffers
    std::vector<double> result;
    auto max_slot = cuts_.back();
    auto array_size = 2 * max_slot * sizeof(double);
    double *slots = static_cast<double *>(malloc(array_size));
    while (remaining > kPrefixLen) {
        DamDecoder decoder(reinterpret_cast<uint8_t *>(pointer), remaining);
        if (!decoder.IsValid()) {
            cout << "Not DAM encoded buffer ignored at offset: "
                 << static_cast<int>((pointer - reinterpret_cast<char *>(buffer))) << endl;
            break;
        }
        auto size = decoder.Size();
        auto node_list = decoder.DecodeIntArray();
        for (auto node : node_list) {
            memset(slots, 0, array_size);
            auto feature_list = decoder.DecodeIntArray();
            // Convert per-feature histo to a flat one
            for (auto f : feature_list) {
                auto base = cuts_[f];
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

void *NVFlareProcessor::ProcessHistograms(size_t *size, const std::vector<double>& histograms) {
    cout << "ProcessHistograms called with " << histograms.size() << " entries" << endl;

    DamEncoder encoder(kDataSetHistograms);
    encoder.AddFloatArray(histograms);
    return encoder.Finish(*size);
}

std::vector<double> NVFlareProcessor::HandleHistograms(void *buffer, size_t buf_size) {
    cout << "HandleHistograms called with buffer size: " << buf_size << endl;

    DamDecoder decoder(reinterpret_cast<uint8_t *>(buffer), buf_size);
    if (!decoder.IsValid()) {
        cout << "Not DAM encoded buffer, ignored" << endl;
        return std::vector<double>();
    }

    if (decoder.GetDataSetId() != kDataSetHistogramResult) {
        cout << "Invalid dataset: " << decoder.GetDataSetId() << endl;
        return std::vector<double>();
    }

    return decoder.DecodeFloatArray();
}

extern "C" {

processing::Processor *LoadProcessor(char *plugin_name) {
    if (strcasecmp(plugin_name, kPluginName) != 0) {
        cout << "Unknown plugin name: " << plugin_name << endl;
        return nullptr;
    }

    return new NVFlareProcessor();
}

}  // extern "C"
