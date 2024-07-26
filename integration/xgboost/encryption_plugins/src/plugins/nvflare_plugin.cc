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
#include <iostream>
#include <algorithm>   // for copy_n, transform
#include <cstring>     // for memcpy
#include <stdexcept>   // for invalid_argument
#include <vector>      // for vector

#include "nvflare_plugin.h"
#include "data_set_ids.h"
#include "dam.h"       // for DamEncoder

namespace nvflare {

void NvflarePlugin::EncryptGPairs(float const *in_gpair, std::size_t n_in,
                                  std::uint8_t **out_gpair,
                                  std::size_t *n_out) {
  if (debug_) {
    std::cout << Ident() << " NvflarePlugin::EncryptGPairs called with pairs size: " << n_in<< std::endl;
  }

  auto pairs = std::vector<float>(in_gpair, in_gpair + n_in);
  gh_pairs_ = std::vector<double>(pairs.cbegin(), pairs.cend());

  DamEncoder encoder(kDataSetGHPairs, false, dam_debug_);
  encoder.AddFloatArray(gh_pairs_);
  std::size_t size;
  auto buffer = encoder.Finish(size);
  if (!out_gpair) {
    throw std::invalid_argument{"Invalid pointer to output gpair."};
  }
  buffer_.resize(size);
  std::copy_n(buffer, size, buffer_.begin());
  free(buffer);
  *out_gpair = buffer_.data();
  *n_out = size;
}

void NvflarePlugin::SyncEncryptedGPairs(std::uint8_t const *in_gpair,
                                        std::size_t n_bytes,
                                        std::uint8_t const **out_gpair,
                                        std::size_t *out_n_bytes) {
  if (debug_) {
    std::cout << Ident() << " NvflarePlugin::SyncEncryptedGPairs called with buffer size: " << n_bytes << std::endl;
  }

  // For NVFlare plugin, nothing needs to be done here
  *out_n_bytes = n_bytes;
  *out_gpair = in_gpair;
}

void NvflarePlugin::ResetHistContext(std::uint32_t const *cutptrs,
                                     std::size_t cutptr_len,
                                     std::int32_t const *bin_idx,
                                     std::size_t n_idx) {
  if (debug_) {
    std::cout << Ident() << " NvFlarePlugin::ResetHistContext called with cutptrs size: " << cutptr_len << " bin_idx size: "
              << n_idx<< std::endl;
  }

  cut_ptrs_.resize(cutptr_len);
  std::copy_n(cutptrs, cutptr_len, cut_ptrs_.begin());
  bin_idx_.resize(n_idx);
  std::copy_n(bin_idx, n_idx, this->bin_idx_.begin());
}

void NvflarePlugin::BuildEncryptedHistVert(std::uint64_t const **ridx,
                                           std::size_t const *sizes,
                                           std::int32_t const *nidx,
                                           std::size_t len,
                                           std::uint8_t** out_hist,
                                           std::size_t* out_len) {
  if (debug_) {
    std::cout << Ident() << " NvflarePlugin::BuildEncryptedHistVert called with len: " << len << std::endl;
  }

  std::int64_t data_set_id;
  if (!feature_sent_) {
    data_set_id = kDataSetAggregationWithFeatures;
    feature_sent_ = true;
  } else {
    data_set_id = kDataSetAggregation;
  }

  DamEncoder encoder(data_set_id, false, dam_debug_);

  // Add cuts pointers
  std::vector<int64_t> cuts_vec(cut_ptrs_.cbegin(), cut_ptrs_.cend());
  encoder.AddIntArray(cuts_vec);

  auto num_features = cut_ptrs_.size() - 1;
  auto num_samples = bin_idx_.size() / num_features;
  if (debug_) {
    std::cout << "Samples: " << num_samples << " Features: " << num_features << std::endl;
  }

  std::vector<int64_t> bins;
  if (data_set_id == kDataSetAggregationWithFeatures) {
    if (features_.empty()) { // when is it not empty?
      for (int64_t f = 0; f < num_features; f++) {
        auto slot = bin_idx_[f];
        if (slot >= 0) {
          // what happens if it's missing?
          features_.push_back(f);
        }
      }
    }
    encoder.AddIntArray(features_);
    
    for (int i = 0; i < num_samples; i++) {
      for (auto f : features_) {
        auto index = f + i * num_features;
        if (index > bin_idx_.size()) {
          throw std::out_of_range{"Index is out of range: " +
                                  std::to_string(index)};
        }
        auto slot = bin_idx_[index];
        bins.push_back(slot);
      }
    }
    encoder.AddIntArray(bins);
  }

  // Add nodes to build
  std::vector<int64_t> node_vec(len);
  for (std::size_t i = 0; i < len; i++) {
    node_vec[i] = nidx[i];
  }
  encoder.AddIntArray(node_vec);

  // For each node, get the row_id/slot pair
  auto row_ids = std::vector<std::vector<int64_t>>(len);
  for (std::size_t i = 0; i < len; ++i) {
    auto& rows = row_ids[i];
    rows.resize(sizes[i]);
    for (std::size_t j = 0; j < sizes[i]; j++) {
      rows[j] = static_cast<int64_t>(ridx[i][j]);
    }
    encoder.AddIntArray(rows);
  }

  std::size_t n{0};
  auto buffer = encoder.Finish(n);
  if (debug_) {
    std::cout << "Finished size:  " << n << std::endl;
  }

  // XGBoost doesn't allow the change of allgatherV sizes. Make sure it's big
  // enough to carry histograms
  auto max_slot = cut_ptrs_.back();
  auto histo_size = 2 * max_slot * sizeof(double) * len + 1024*1024; // 1M is DAM overhead
  auto buf_size = histo_size > n ? histo_size : n;

  // Copy to an array so the buffer can be freed, should change encoder to return vector
  buffer_.resize(buf_size);
  std::copy_n(buffer, n, buffer_.begin());
  free(buffer);

  *out_hist = buffer_.data();
  *out_len = buffer_.size();
}

void NvflarePlugin::SyncEncryptedHistVert(std::uint8_t *buffer,
                                          std::size_t buf_size,
                                          double **out,
                                          std::size_t *out_len) {
  if (debug_) {
    std::cout << Ident() << " NvflarePlugin::SyncEncryptedHistVert called with buffer size: " << buf_size << std::endl;
  }

  auto remaining = buf_size;
  char *pointer = reinterpret_cast<char *>(buffer);

  // The buffer is concatenated by AllGather. It may contain multiple DAM buffers
  std::vector<double> &result = histo_;
  result.clear();
  auto max_slot = cut_ptrs_.back();
  auto array_size = 2 * max_slot * sizeof(double);

  // A new histogram array?
  auto slots = static_cast<double *>(malloc(array_size));
  while (remaining > kPrefixLen) {
    DamDecoder decoder(reinterpret_cast<uint8_t *>(pointer), remaining, false, dam_debug_);
    if (!decoder.IsValid()) {
      std::cout << "Not DAM encoded buffer ignored at offset: "
                << static_cast<int>((pointer - reinterpret_cast<char *>(buffer))) << std::endl;
      break;
    }
    auto size = decoder.Size();
    auto node_list = decoder.DecodeIntArray();
    if (debug_) {
      std::cout << "Number of nodes: " << node_list.size() << " Histo size: " << 2*max_slot << std::endl;
    }
    for ([[maybe_unused]] auto node : node_list) {
      std::memset(slots, 0, array_size);
      auto feature_list = decoder.DecodeIntArray();
      // Convert per-feature histo to a flat one
      for (auto f : feature_list) {
        auto base = cut_ptrs_[f]; // cut pointer for the current feature
        auto bins = decoder.DecodeFloatArray();
        auto n = bins.size() / 2;
        for (int i = 0; i < n; i++) {
          auto index = base + i;
          // [Q] Build local histogram? Why does it need to be built here?
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

  // result is a reference to a histo_
  *out_len = result.size();
  *out = result.data();
  if (debug_) {
    std::cout << "Total histogram size: " << *out_len << std::endl;
  }
}

void NvflarePlugin::BuildEncryptedHistHori(double const *in_histogram,
                                           std::size_t len,
                                           std::uint8_t **out_hist,
                                           std::size_t *out_len) {
  if (debug_) {
    std::cout << Ident() << " NvflarePlugin::BuildEncryptedHistHori called with histo size: " << len << std::endl;
  }

  DamEncoder encoder(kDataSetHistograms, false, dam_debug_);
  std::vector<double> copy(in_histogram, in_histogram + len);
  encoder.AddFloatArray(copy);

  std::size_t size{0};
  auto buffer = encoder.Finish(size);
  buffer_.resize(size);
  std::copy_n(buffer, size, buffer_.begin());
  free(buffer);

  *out_hist = this->buffer_.data();
  *out_len = this->buffer_.size();
}

void NvflarePlugin::SyncEncryptedHistHori(std::uint8_t const *buffer,
                                          std::size_t len,
                                          double **out_hist,
                                          std::size_t *out_len) {
  if (debug_) {
    std::cout << Ident() << " NvflarePlugin::SyncEncryptedHistHori called with buffer size: " << len << std::endl;
  }

  auto remaining = len;
  auto pointer = buffer;

  // The buffer is concatenated by AllGather. It may contain multiple DAM buffers
  std::vector<double>& result = histo_;
  result.clear();
  while (remaining > kPrefixLen) {
    DamDecoder decoder(const_cast<std::uint8_t *>(pointer), remaining, false, dam_debug_);
    if (!decoder.IsValid()) {
      std::cout << "Not DAM encoded histogram ignored at offset: "
                << static_cast<int>(pointer - buffer) << std::endl;
      break;
    }

    if (decoder.GetDataSetId() != kDataSetHistogramResult) {
      throw std::runtime_error{"Invalid dataset: " + std::to_string(decoder.GetDataSetId())};
    }

    auto size = decoder.Size();
    auto histo = decoder.DecodeFloatArray();
    result.insert(result.end(), histo.cbegin(), histo.cend());

    remaining -= size;
    pointer += size;
  }

  *out_hist = result.data();
  *out_len = result.size();
}

} // namespace nvflare
