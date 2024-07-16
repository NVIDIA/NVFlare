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
#include "xgboost_plugin.h"

namespace nvflare
{
    // A pass-through plugin that doesn't encrypt any data
    class PassThruPlugin : public BasePlugin
    {
    public:
        explicit PassThruPlugin(std::vector<std::pair<std::string_view, std::string_view>> const& args) : BasePlugin(
            args)
        {
        }

        ~PassThruPlugin() override = default;

        void EncryptGPairs(const float* in_gpair, std::size_t n_in, std::uint8_t** out_gpair,
                           std::size_t* n_out) override;

        void SyncEncryptedGPairs(const std::uint8_t* in_gpair, std::size_t n_bytes, const std::uint8_t** out_gpair,
                                 std::size_t* out_n_bytes) override;

        void ResetHistContext(const std::uint32_t* cutptrs, std::size_t cutptr_len, const std::int32_t* bin_idx,
                              std::size_t n_idx) override;

        void BuildEncryptedHistHori(const double* in_histogram, std::size_t len, std::uint8_t** out_hist,
                                    std::size_t* out_len) override;

        void SyncEncryptedHistHori(const std::uint8_t* buffer, std::size_t len, double** out_hist,
                                   std::size_t* out_len) override;

        void BuildEncryptedHistVert(const std::size_t** ridx, const std::size_t* sizes, const std::int32_t* nidx,
                                    std::size_t len, std::uint8_t** out_hist, std::size_t* out_len) override;

        void SyncEncryptedHistVert(std::uint8_t* hist_buffer, std::size_t len, double** out,
                                   std::size_t* out_len) override;
    };
} // namespace nvflare
