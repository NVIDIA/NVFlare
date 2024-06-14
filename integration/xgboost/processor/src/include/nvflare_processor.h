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
#include <cstdint>     // for uint8_t, uint32_t, int32_t, int64_t
#include <string_view> // for string_view
#include <utility>     // for pair
#include <vector>      // for vector

const int kDataSetHGPairs = 1;
const int kDataSetAggregation = 2;
const int kDataSetAggregationWithFeatures = 3;
const int kDataSetAggregationResult = 4;
const int kDataSetHistograms = 5;
const int kDataSetHistogramResult = 6;

// Opaque pointer type for the C API.
typedef void *FederatedPluginHandle; // NOLINT

namespace nvflare {
// Plugin that uses Python tenseal and GRPC.
class TensealPlugin {
  // Buffer for storing encrypted gradient pairs.
  std::vector<std::uint8_t> encrypted_gpairs_;
  // Buffer for histogram cut pointers (indptr of a CSC).
  std::vector<std::uint32_t> cut_ptrs_;
  // Buffer for histogram index.
  std::vector<std::int32_t> bin_idx_;

  bool feature_sent_{false};
  // The feature index.
  std::vector<std::int64_t> features_;
  // Buffer for output histogram.
  std::vector<std::uint8_t> encrypted_hist_;
  std::vector<double> hist_;

public:
  TensealPlugin(
      std::vector<std::pair<std::string_view, std::string_view>> const &args);
  // Gradient pairs
  void EncryptGPairs(float const *in_gpair, std::size_t n_in,
                     std::uint8_t **out_gpair, std::size_t *n_out);
  void SyncEncryptedGPairs(std::uint8_t const *in_gpair, std::size_t n_bytes,
                           std::uint8_t const **out_gpair,
                           std::size_t *out_n_bytes);

  // Histogram
  void ResetHistContext(std::uint32_t const *cutptrs, std::size_t cutptr_len,
                        std::int32_t const *bin_idx, std::size_t n_idx);
  void BuildEncryptedHistHori(double const *in_histogram, std::size_t len,
                              std::uint8_t **out_hist, std::size_t *out_len);
  void SyncEncryptedHistHori(std::uint8_t const *buffer, std::size_t len,
                             double **out_hist, std::size_t *out_len);

  void BuildEncryptedHistVert(std::size_t const **ridx,
                              std::size_t const *sizes,
                              std::int32_t const *nidx, std::size_t len,
                              std::uint8_t **out_hist, std::size_t *out_len);
  void SyncEncryptedHistVert(std::uint8_t *hist_buffer, std::size_t len,
                             double **out, std::size_t *out_len);
};
} // namespace nvflare
