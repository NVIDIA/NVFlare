/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#include <iostream>
#include <cstring>
#include <cstdint>
#include <chrono>
#include "local_processor.h"
#include "data_set_ids.h"
#include "util.h"

const char kParamDebug[] = "debug";
const char kParamDamDebug[] = "dam_debug";
const char kParamPrintTiming[] = "print_timing";

void LocalProcessor::Initialize(bool active, std::map<std::string, std::string> params) {
    active_ = active;
    print_timing_ = get_bool(params, kParamPrintTiming);
    debug_ = get_bool(params, kParamDebug);
    dam_debug_ = get_bool(params, kParamDamDebug);
}

void LocalProcessor::Shutdown() {
    gh_pairs_.clear();
    FreeEncryptedData(encrypted_gh_);
    histo_.clear();
    cuts_.clear();
    slots_.clear();
}

void LocalProcessor::FreeBuffer(void *buffer) {
    free(buffer);
}

void* LocalProcessor::ProcessGHPairs(std::size_t *size, const std::vector<double>& pairs) {
    if (debug_) {
        std::cout << "ProcessGHPairs called with pairs size: " << pairs.size() << std::endl;
    }

    if (print_timing_) {
        std::cout << "Encrypting " << pairs.size()/2 << " GH Pairs" << std::endl;
    }
    auto start = std::chrono::system_clock::now();

    auto encrypted_data = EncryptVector(pairs);

    if (print_timing_) {
        auto end = std::chrono::system_clock::now();
        auto secs = (double) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
        std::cout << "Encryption time: " << secs << " seconds" << std::endl;
    }

    DamEncoder encoder(kDataSetGHPairs, true, dam_debug_);
    encoder.AddBuffer(encrypted_data);
    auto buffer = encoder.Finish(*size);
    FreeEncryptedData(encrypted_data);

    // Save pairs for future operations. This is only called on active site
    gh_pairs_ = std::vector<double>(pairs);

    return buffer;
}

void* LocalProcessor::HandleGHPairs(std::size_t *size, void *buffer, std::size_t buf_size) {
    if (debug_) {
        std::cout << "HandleGHPairs called with buffer size: " << buf_size << " Active: " << active_ << std::endl;
    }

    *size = buf_size;

    if (active_) {
        // Do nothing for active site
        return buffer;
    }

    auto decoder = DamDecoder(reinterpret_cast<std::uint8_t *>(buffer), buf_size, true, dam_debug_);
    if (!decoder.IsValid()) {
        return buffer;
    }

    auto encrypted_buffer = decoder.DecodeBuffer();
    if (debug_) {
        std::cout << "Encrypted buffer size: " << encrypted_buffer.buf_size << std::endl;
    }

    // The caller may free buffer so a copy is needed
    FreeEncryptedData(encrypted_gh_);
    auto buf = malloc(encrypted_buffer.buf_size);
    memcpy(buf, encrypted_buffer.buffer, encrypted_buffer.buf_size);
    encrypted_gh_ = Buffer(buf, encrypted_buffer.buf_size, true);
    FreeEncryptedData(encrypted_buffer);

    return buffer;
}

void LocalProcessor::InitAggregationContext(const std::vector<uint32_t> &cuts, const std::vector<int> &slots) {
    if (this->slots_.empty()) {
        this->cuts_ = std::vector<uint32_t>(cuts);
        this->slots_ = std::vector<int>(slots);
    } else {
        std::cout << "Multiple calls to InitAggregationContext" << std::endl;
    }
}

void *LocalProcessor::ProcessAggregation(std::size_t *size, std::map<int, std::vector<int>> nodes) {
    if (debug_) {
        std::cout << "ProcessAggregation called with " << nodes.size() << " nodes" << std::endl;
    }

    void *result;

    if (active_) {
        result = ProcessClearAggregation(size, nodes);
    } else {
        result = ProcessEncryptedAggregation(size, nodes);
    }

    // print_buffer(reinterpret_cast<uint8_t *>(result), *size);

    return result;
}

void *LocalProcessor::ProcessClearAggregation(std::size_t *size, std::map<int, std::vector<int>>& nodes) {
    if (debug_) {
        std::cout << "ProcessClearAggregation called with " << nodes.size() << " nodes" << std::endl;
    }

    auto total_bin_size = cuts_.back();
    auto histo_size = total_bin_size*2;
    auto total_size = histo_size * nodes.size();

    histo_.clear();
    histo_.resize(total_size, 0.0);
    size_t start = 0;
    for (const auto &node : nodes) {
        auto rows = node.second;
        for (const auto &row_id : rows) {
            auto num = cuts_.size() - 1;
            for (std::size_t f = 0; f < num; f++) {
                int slot = slots_[f + num * row_id];
                if ((slot < 0) || (slot >= total_bin_size)) {
                    continue;
                }
                auto g = (gh_pairs_)[row_id * 2];
                auto h = (gh_pairs_)[row_id * 2 + 1];
                (histo_)[start + slot * 2] += g;
                (histo_)[start + slot * 2 + 1] += h;
            }
        }
        start += histo_size;
    }

    // Histogram is in clear, can't send to all_gather. Just return empty DAM buffer
    auto encoder = DamEncoder(kDataSetAggregationResult, true, dam_debug_);
    encoder.AddBuffer(Buffer());
    return encoder.Finish(*size);
}

void *LocalProcessor::ProcessEncryptedAggregation(std::size_t *size, std::map<int, std::vector<int>>& nodes) {
    if (debug_) {
        std::cout << "ProcessEncryptedAggregation called with " << nodes.size() << " nodes" << std::endl;
    }

    auto num_slot = cuts_.back();
    auto total_size = num_slot * nodes.size();

    auto encrypted_histo = std::vector<Buffer>(total_size);
    size_t offset = 0;
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
                auto &row_ids = row_id_map[slot];
                row_ids.push_back(row_id);
            }
        }

        if (print_timing_) {
            int add_ops = 0;
            for (auto &item : row_id_map) {
                add_ops += item.second.size();
            }
            std::cout << "Aggregating with " << add_ops << " additions" << std::endl;
        }
        auto start = std::chrono::system_clock::now();

        auto encrypted_sum = AddGHPairs(row_id_map);

        if (print_timing_) {
            auto end = std::chrono::system_clock::now();
            auto secs = (double) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
            std::cout << "Aggregation time: " << secs << " seconds" << std::endl;
        }

        // Convert map back to array
        for (int slot = 0; slot < num_slot; slot++) {
            auto it = encrypted_sum.find(slot);
            if (it != encrypted_sum.end()) {
                encrypted_histo[offset + slot] = it->second;
            }
        }

        offset += num_slot;
    }

    auto encoder = DamEncoder(kDataSetAggregationResult, true, dam_debug_);
    encoder.AddBufferArray(encrypted_histo);
    auto result = encoder.Finish(*size);

    for (auto& item : encrypted_histo) {
        FreeEncryptedData(item);
    }

    return result;
}

std::vector<double> LocalProcessor::HandleAggregation(void *buffer, std::size_t buf_size) {
    if (debug_) {
        std::cout << "HandleAggregation called with buffer size: " << buf_size
                  << " Active: " << active_ << std::endl;
    }

    auto remaining = buf_size;
    auto pointer = reinterpret_cast<uint8_t *>(buffer);

    std::vector<double> result;

    if (!active_) {
        if (debug_) {
            std::cout << "Result size: " << result.size() << std::endl;
        }
        return result;
    }

    // The buffer is concatenated by AllGather. It may contain multiple DAM buffers
    auto first = true;
    while (remaining > kPrefixLen) {
        DamDecoder decoder(pointer, remaining, true, dam_debug_);
        if (!decoder.IsValid()) {
            std::cout << "Not DAM encoded buffer ignored at offset: "
                 << static_cast<int>((pointer - reinterpret_cast<uint8_t *>(buffer))) << std::endl;
            break;
        }
        auto size = decoder.Size();
        if (first) {
            if (histo_.empty()) {
                std::cout << "No clear histogram." << std::endl;
                return result;
            }
            result.insert(result.end(), histo_.begin(), histo_.end());
            first = false;
        } else {
            auto encrypted_buf = decoder.DecodeBufferArray();

            if (print_timing_) {
                std::cout << "Decrypting " << encrypted_buf.size() << " pairs" << std::endl;
            }
            auto start = std::chrono::system_clock::now();

            auto decrypted_histo = DecryptVector(encrypted_buf);

            if (print_timing_) {
                auto end = std::chrono::system_clock::now();
                auto secs = (double) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
                std::cout << "Decryption time: " << secs << " seconds" << std::endl;
            }

            if (decrypted_histo.size() != histo_.size()) {
                std::cout << "Histo sizes are different: " << decrypted_histo.size()
                    << " != " <<  histo_.size()  << std::endl;
            }
            result.insert(result.end(), decrypted_histo.begin(), decrypted_histo.end());
        }
        remaining -= size;
        pointer += size;
    }

    if (debug_) {
        std::cout << "Decrypted result size: " << result.size() << std::endl;
    }

    // print_buffer(reinterpret_cast<uint8_t *>(result.data()), result.size()*8);

    return result;
}

// Horizontal encryption is still handled by NVFlare so those two methods uses normal, not local signature
void *LocalProcessor::ProcessHistograms(size_t *size, const std::vector<double>& histograms) {
    if (debug_) {
        std::cout << "Remote ProcessHistograms called with " << histograms.size() << " entries" << std::endl;
    }

    DamEncoder encoder(kDataSetHistograms, false, dam_debug_);
    encoder.AddFloatArray(histograms);
    return encoder.Finish(*size);
}

std::vector<double> LocalProcessor::HandleHistograms(void *buffer, size_t buf_size) {
    if (debug_) {
        std::cout << "Remote HandleHistograms called with buffer size: " << buf_size << std::endl;
    }

    DamDecoder decoder(reinterpret_cast<uint8_t *>(buffer), buf_size, false, dam_debug_);
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
