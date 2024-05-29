/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#include <iostream>
#include <cstring>
#include <cstdint>
#include "local_processor.h"
#include "dam.h"

void* LocalProcessor::ProcessGHPairs(std::size_t *size, const std::vector<double>& pairs) {
    auto encrypted_data = EncryptVector(pairs);

    DamEncoder encoder(kDataSetGHPairs, true);
    encoder.AddBytes(encrypted_data.buffer, encrypted_data.buf_size);
    auto buffer = encoder.Finish(*size);
    FreeEncryptedData(encrypted_data);

    // Save pairs for future operations. This is only called on active site
    this->gh_pairs_ = new std::vector<double>(pairs);

    return buffer;
}

void* LocalProcessor::HandleGHPairs(std::size_t *size, void *buffer, std::size_t buf_size) {
    *size = buf_size;

    if (active_) {
        // Do nothing for active site
        return buffer;
    }

    auto decoder = DamDecoder(reinterpret_cast<std::uint8_t *>(buffer), buf_size, true);
    if (!decoder.IsValid()) {
        return buffer;
    }

    size_t encrypted_size;
    auto encrypted_buffer = decoder.DecodeBytes(&encrypted_size);

    // The caller may free buffer so a copy is needed
    if (encrypted_gh_.buffer) {
        free(encrypted_gh_.buffer);
    }
    encrypted_gh_.buffer = malloc(encrypted_size);
    memcpy(encrypted_gh_.buffer, encrypted_buffer, encrypted_size);
    encrypted_gh_.buf_size = encrypted_size;

    return buffer;
}

void *LocalProcessor::ProcessAggregation(std::size_t *size, std::map<int, std::vector<int>> nodes) {
    void *result;

    if (active_) {
        result = ProcessClearAggregation(size, nodes);
    } else {
        result = ProcessEncryptedAggregation(size, nodes);
    }

    return result;
}

void *LocalProcessor::ProcessClearAggregation(std::size_t *size, std::map<int, std::vector<int>> nodes) {
    int total_bin_size = cuts_.back();
    int histo_size = total_bin_size*2;
    int total_size = histo_size * nodes.size();

    histo_ = new std::vector<double>(total_size);
    int start = 0;
    for (const auto &node : nodes) {
        auto rows = node.second;
        for (const auto &row_id : rows) {
            auto num = cuts_.size() - 1;
            for (std::size_t f = 0; f < num; f++) {
                int slot = slots_[f + num * row_id];
                if ((slot < 0) || (slot >= total_bin_size)) {
                    continue;
                }
                auto g = (*gh_pairs_)[row_id * 2];
                auto h = (*gh_pairs_)[row_id * 2 + 1];
                (*histo_)[start + slot * 2] += g;
                (*histo_)[start + slot * 2 + 1] += h;
            }
        }
        start += histo_size;
    }

    // Histogram is in clear, can't send to all_gather. Just return empty DAM buffer
    auto encoder = DamEncoder(kDataSetAggregationResult, true);
    encoder.AddBytes(nullptr, 0);
    return encoder.Finish(*size);
}

void *LocalProcessor::ProcessEncryptedAggregation(std::size_t *size, std::map<int, std::vector<int>> nodes) {
    int num_slot = cuts_.back();
    int total_size = num_slot * nodes.size();

    auto encrypted_histo = new std::vector<Buffer>(total_size);
    int start = 0;
    for (const auto &node : nodes) {
        auto rows = node.second;
        auto num = cuts_.size() - 1;
        auto row_id_map = std::map<int, std::vector<int>>();
        for (std::size_t f = 0; f < num; f++) {
            for (const auto &row_id : rows) {
                int slot = slots_[f + num * row_id];
                if ((slot < 0) || (slot >= num_slot)) {
                    continue;
                }
                auto row_ids = row_id_map[slot];
                row_ids.push_back(row_id);
            }
        }

        auto encrypted_sum = AddGHPairs(row_id_map);
        for (int slot = 0; slot < num_slot; slot++) {
            auto it = encrypted_sum.find(slot);
            if (it != encrypted_sum.end()) {
                encrypted_histo[start + slot] = it->second;
            }
        }

        start += num_slot;
    }

    // Histogram is in clear, can't send to all_gather. Just return empty DAM buffer
    auto encoder = DamEncoder(kDataSetAggregationResult, true);
    encoder.AddBytesArray(encrypted_histo);
    return encoder.Finish(*size);
}


std::vector<double> LocalProcessor::HandleAggregation(void *buffer, std::size_t buf_size) {
    std::vector<double> result = std::vector<double>();

    int8_t* ptr = static_cast<int8_t *>(buffer);
    auto rest_size = buf_size;

    while (rest_size > kPrefixLen) {
        if (!ValidDam(ptr, rest_size)) {
            break;
        }
        int64_t *size_ptr = reinterpret_cast<int64_t *>(ptr + 8);
        double *array_start = reinterpret_cast<double *>(ptr + kPrefixLen);
        auto array_size = (*size_ptr - kPrefixLen)/8;
        result.insert(result.end(), array_start, array_start + array_size);
        rest_size -= *size_ptr;
        ptr = ptr + *size_ptr;
    }

    double *histo = reinterpret_cast<double *>(buf + kPrefixLen);
    for ( const auto &node : nodes ) {
        auto rows = node.second;
        for (const auto &row_id : rows) {
            auto num = cuts_.size() - 1;
            for (std::size_t f = 0; f < num; f++) {
                int slot = slots_[f + num*row_id];
                if ((slot < 0) || (slot >= total_bin_size)) {
                    continue;
                }

                auto g = (*gh_pairs_)[row_id*2];
                auto h = (*gh_pairs_)[row_id*2+1];
                histo[slot*2] += g;
                histo[slot*2+1] += h;
            }
        }
        histo += histo_size;
    }


    return result;
}

void* LocalProcessor::ProcessHistograms(std::size_t *size, const std::vector<double>& histograms) {
    *size = kPrefixLen + histograms.size()*10*8;  // Assume encrypted size is 10x

    int64_t buf_size = *size;
    // This memory needs to be freed
    char *buf = static_cast<char *>(malloc(buf_size));
    memcpy(buf, kSignature, strlen(kSignature));
    memcpy(buf + 8, &buf_size, 8);
    memcpy(buf + 16, &kDataTypeAggregatedHisto, 8);

    // Simulate encryption by duplicating value 10 times
    int index = kPrefixLen;
    for (auto value : histograms) {
        for (std::size_t i = 0; i < 10; i++) {
            memcpy(buf+index, &value, 8);
            index += 8;
        }
    }

    return buf;
}

std::vector<double> LocalProcessor::HandleHistograms(void *buffer, std::size_t buf_size) {
    std::vector<double> result = std::vector<double>();

    int8_t* ptr = static_cast<int8_t *>(buffer);
    auto rest_size = buf_size;

    while (rest_size > kPrefixLen) {
        if (!ValidDam(ptr, rest_size)) {
            break;
        }
        int64_t *size_ptr = reinterpret_cast<int64_t *>(ptr + 8);
        double *array_start = reinterpret_cast<double *>(ptr + kPrefixLen);
        auto array_size = (*size_ptr - kPrefixLen)/8;
        auto empty = result.empty();
        if (!empty) {
            if (result.size() != array_size / 10) {
                std::cout << "Histogram size doesn't match " << result.size()
                          << " != " << array_size << std::endl;
                return result;
            }
        }

        for (std::size_t i = 0; i < array_size/10; i++) {
            auto value = array_start[i*10];
            if (empty) {
                result.push_back(value);
            } else {
                result[i] += value;
            }
        }

        rest_size -= *size_ptr;
        ptr = ptr + *size_ptr;
    }

    return result;
}

