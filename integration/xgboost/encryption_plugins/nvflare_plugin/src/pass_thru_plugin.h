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
#include "local_plugin.h"

namespace nvflare {
  // A pass-through plugin that doesn't encrypt any data
  class PassThruPlugin : public LocalPlugin {
  public:
    explicit PassThruPlugin(std::vector<std::pair<std::string_view, std::string_view>> const &args) :
        LocalPlugin(args) {}

    ~PassThruPlugin() override = default;

    // Horizontal in local plugin still goes through NVFlare, so it needs to be overwritten
    void BuildEncryptedHistHori(const double *in_histogram, std::size_t len, std::uint8_t **out_hist,
                                std::size_t *out_len) override;

    void SyncEncryptedHistHori(const std::uint8_t *buffer, std::size_t len, double **out_hist,
                               std::size_t *out_len) override;

    Buffer EncryptVector(const std::vector<double> &cleartext) override;

    std::vector<double> DecryptVector(const std::vector<Buffer> &ciphertext) override;

    void AddGHPairs(std::vector<Buffer>& result, const std::uint64_t *ridx, const std::size_t size) override;

  };
} // namespace nvflare
