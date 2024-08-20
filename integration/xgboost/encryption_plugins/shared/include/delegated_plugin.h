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
#include "base_plugin.h"

namespace nvflare {

// Plugin that delegates to other real plugins
class DelegatedPlugin : public BasePlugin {

  BasePlugin *plugin_{nullptr};

public:
  explicit DelegatedPlugin(std::vector<std::pair<std::string_view, std::string_view>> const &args);

  ~DelegatedPlugin() override {
      delete plugin_;
  }

  void EncryptGPairs(const float* in_gpair, std::size_t n_in, std::uint8_t** out_gpair, std::size_t* n_out) override {
    plugin_->EncryptGPairs(in_gpair, n_in, out_gpair, n_out);
  }

  void SyncEncryptedGPairs(const std::uint8_t* in_gpair, std::size_t n_bytes, const std::uint8_t** out_gpair,
    std::size_t* out_n_bytes) override {
    plugin_->SyncEncryptedGPairs(in_gpair, n_bytes, out_gpair, out_n_bytes);
  }

  void ResetHistContext(const std::uint32_t* cutptrs, std::size_t cutptr_len, const std::int32_t* bin_idx,
    std::size_t n_idx) override {
    plugin_->ResetHistContext(cutptrs, cutptr_len, bin_idx, n_idx);
  }

  void BuildEncryptedHistHori(const double* in_histogram, std::size_t len, std::uint8_t** out_hist,
    std::size_t* out_len) override {
    plugin_->BuildEncryptedHistHori(in_histogram, len, out_hist, out_len);
  }

  void SyncEncryptedHistHori(const std::uint8_t* buffer, std::size_t len, double** out_hist,
    std::size_t* out_len) override {
    plugin_->SyncEncryptedHistHori(buffer, len, out_hist, out_len);
  }

  void BuildEncryptedHistVert(const std::uint64_t** ridx, const std::size_t* sizes, const std::int32_t* nidx,
    std::size_t len, std::uint8_t** out_hist, std::size_t* out_len) override {
    plugin_->BuildEncryptedHistVert(ridx, sizes, nidx, len, out_hist, out_len);
  }

  void SyncEncryptedHistVert(std::uint8_t* hist_buffer, std::size_t len, double** out, std::size_t* out_len) override {
    plugin_->SyncEncryptedHistVert(hist_buffer, len, out, out_len);
  }
};
} // namespace nvflare
