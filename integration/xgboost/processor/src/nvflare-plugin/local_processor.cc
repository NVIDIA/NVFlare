/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#include <iostream>
#include <cstring>
#include <cstdint>
#include "local_processor.h"
#include "data_set_ids.h"

void* LocalProcessor::ProcessGHPairs(std::size_t *size, const std::vector<double>& pairs) {
    std::cout << "ProcessGHPairs called with pairs size: " << pairs.size() << std::endl;
    auto encrypted_data = EncryptVector(pairs);

    DamEncoder encoder(kDataSetGHPairs, true);
    encoder.AddBuffer(encrypted_data);
    auto buffer = encoder.Finish(*size);
    FreeEncryptedData(encrypted_data);

    // Save pairs for future operations. This is only called on active site
    this->gh_pairs_ = new std::vector<double>(pairs);

    return buffer;
}

void* LocalProcessor::HandleGHPairs(std::size_t *size, void *buffer, std::size_t buf_size) {
    std::cout << "HandleGHPairs called with buffer size: " << buf_size << " Active: " << active_ << std::endl;
    *size = buf_size;

    if (active_) {
        // Do nothing for active site
        return buffer;
    }

    auto decoder = DamDecoder(reinterpret_cast<std::uint8_t *>(buffer), buf_size, true);
    if (!decoder.IsValid()) {
        return buffer;
    }

    auto encrypted_buffer = decoder.DecodeBuffer();

    // The caller may free buffer so a copy is needed
    if (encrypted_gh_.buffer) {
        free(encrypted_gh_.buffer);
    }
    encrypted_gh_.buffer = malloc(encrypted_buffer.buf_size);
    memcpy(encrypted_gh_.buffer, encrypted_buffer.buffer, encrypted_buffer.buf_size);
    encrypted_gh_.buf_size = encrypted_buffer.buf_size;

    return buffer;
}

void *LocalProcessor::ProcessAggregation(std::size_t *size, std::map<int, std::vector<int>> nodes) {
    std::cout << "ProcessAggregation called with " << nodes.size() << " nodes" << std::endl;
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
    encoder.AddBuffer(Buffer());
    return encoder.Finish(*size);
}

void *LocalProcessor::ProcessEncryptedAggregation(std::size_t *size, std::map<int, std::vector<int>> nodes) {
    int num_slot = cuts_.back();
    int total_size = num_slot * nodes.size();

    auto encrypted_histo = std::vector<Buffer>(total_size);
    int start = 0;
    for (const auto &node : nodes) {
        auto rows = node.second;
        auto num = cuts_.size() - 1;
        auto row_id_map = std::map<int, std::vector<int>>();

        // Empty slot leaks data so fill everything with empty vectors
        for (int slot = 0; slot < num_slot; slot++) {
            row_id_map.insert({slot, std::vector<int>()});
        }

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

        // Convert map back to array
        for (int slot = 0; slot < num_slot; slot++) {
            auto it = encrypted_sum.find(slot);
            if (it != encrypted_sum.end()) {
                encrypted_histo[start + slot] = it->second;
            }
        }

        start += num_slot;
    }

    auto encoder = DamEncoder(kDataSetAggregationResult, true);
    encoder.AddBufferArray(encrypted_histo);
    return encoder.Finish(*size);
}

std::vector<double> LocalProcessor::HandleAggregation(void *buffer, std::size_t buf_size) {
    std::cout << "HandleAggregation called with buffer size: " << buf_size << std::endl;
    auto remaining = buf_size;
    char *pointer = reinterpret_cast<char *>(buffer);

    // The buffer is concatenated by AllGather. It may contain multiple DAM buffers
    std::vector<double> result;
    auto first = true;
    while (remaining > kPrefixLen) {
        DamDecoder decoder(reinterpret_cast<uint8_t *>(pointer), remaining, true);
        if (!decoder.IsValid()) {
            std::cout << "Not DAM encoded buffer ignored at offset: "
                 << static_cast<int>((pointer - reinterpret_cast<char *>(buffer))) << std::endl;
            break;
        }
        auto size = decoder.Size();
        if (first) {
            result.insert(result.end(), histo_->begin(), histo_->end());
            first = false;
        } else {
            auto encrypted_histo = decoder.DecodeBufferArray();
            auto decrypted_histo = DecryptVector(encrypted_histo);
            result.insert(result.end(), decrypted_histo.begin(), decrypted_histo.end());
        }
        remaining -= size;
        pointer += size;
    }

    return result;
}

// Horizontal encryption is still handled by NVFlare so those two methods uses normal, not local signature
void *LocalProcessor::ProcessHistograms(size_t *size, const std::vector<double>& histograms) {
    std::cout << "Remote ProcessHistograms called with " << histograms.size() << " entries" << std::endl;

    DamEncoder encoder(kDataSetHistograms);
    encoder.AddFloatArray(histograms);
    return encoder.Finish(*size);
}

std::vector<double> LocalProcessor::HandleHistograms(void *buffer, size_t buf_size) {
    std::cout << "Remote HandleHistograms called with buffer size: " << buf_size << std::endl;

    DamDecoder decoder(reinterpret_cast<uint8_t *>(buffer), buf_size);
    if (!decoder.IsValid()) {
        std::cout << "Not DAM encoded buffer, ignored" << std::endl;
        return std::vector<double>();
    }

    if (decoder.GetDataSetId() != kDataSetHistogramResult) {
        std::cout << "Invalid dataset: " << decoder.GetDataSetId() << std::endl;
        return std::vector<double>();
    }

    return decoder.DecodeFloatArray();
}

