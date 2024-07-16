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

#include "util.h"

namespace nvflare {
// Plugin that uses Python tenseal and GRPC.
class BasePlugin {
protected:
  bool debug_ = false;

public:
  explicit BasePlugin(
      std::vector<std::pair<std::string_view, std::string_view>> const &args)
  {
      debug_ = get_bool(args, "debug");
  }

  virtual ~BasePlugin() = default;

  // Gradient pairs
  virtual void EncryptGPairs(float const *in_gpair, std::size_t n_in,
                     std::uint8_t **out_gpair, std::size_t *n_out) = 0;

  virtual void SyncEncryptedGPairs(std::uint8_t const *in_gpair, std::size_t n_bytes,
                           std::uint8_t const **out_gpair,
                           std::size_t *out_n_bytes) = 0;

  // Histogram
  virtual void ResetHistContext(std::uint32_t const *cutptrs, std::size_t cutptr_len,
                        std::int32_t const *bin_idx, std::size_t n_idx) = 0;

  virtual void BuildEncryptedHistHori(double const *in_histogram, std::size_t len,
                              std::uint8_t **out_hist, std::size_t *out_len) = 0;

  virtual void SyncEncryptedHistHori(std::uint8_t const *buffer, std::size_t len,
                             double **out_hist, std::size_t *out_len) = 0;

  virtual void BuildEncryptedHistVert(std::size_t const **ridx,
                              std::size_t const *sizes,
                              std::int32_t const *nidx, std::size_t len,
                              std::uint8_t **out_hist, std::size_t *out_len) = 0;

  virtual void SyncEncryptedHistVert(std::uint8_t *hist_buffer, std::size_t len,
                             double **out, std::size_t *out_len) = 0;
};
} // namespace nvflare
