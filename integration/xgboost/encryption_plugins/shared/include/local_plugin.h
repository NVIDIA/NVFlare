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
#include "dam.h"

namespace nvflare {

// A base plugin for all plugins that handle encryption locally in C++
class LocalPlugin : public BasePlugin {
protected:
  std::vector<double> gh_pairs_;
  std::vector<uint8_t> encrypted_gh_;
  std::vector<double> histo_;
  std::vector<uint32_t> cuts_;
  std::vector<int32_t> bin_idx_vec_;
  std::vector<uint8_t> buffer_;

public:
  explicit LocalPlugin(std::vector<std::pair<std::string_view, std::string_view>> const &args) :
      BasePlugin(args) {}

  ~LocalPlugin() override = default;

  void EncryptGPairs(const float *in_gpair, std::size_t n_in, std::uint8_t **out_gpair,
                     std::size_t *n_out) override;

  void SyncEncryptedGPairs(const std::uint8_t *in_gpair, std::size_t n_bytes, const std::uint8_t **out_gpair,
                           std::size_t *out_n_bytes) override;

  void ResetHistContext(const std::uint32_t *cutptrs, std::size_t cutptr_len, const std::int32_t *bin_idx,
                        std::size_t n_idx) override;

  void BuildEncryptedHistHori(const double *in_histogram, std::size_t len, std::uint8_t **out_hist,
                              std::size_t *out_len) override;

  void SyncEncryptedHistHori(const std::uint8_t *buffer, std::size_t len, double **out_hist,
                             std::size_t *out_len) override;

  void BuildEncryptedHistVert(const std::uint64_t **ridx, const std::size_t *sizes, const std::int32_t *nidx,
                              std::size_t len, std::uint8_t **out_hist, std::size_t *out_len) override;

  void SyncEncryptedHistVert(std::uint8_t *hist_buffer, std::size_t len, double **out,
                             std::size_t *out_len) override;

  // Method needs to be implemented by local plugins

  /*!
   * \brief Encrypt a vector of float-pointing numbers
   * \param cleartext A vector of numbers in cleartext
   * \return A buffer with serialized ciphertext
   */
  virtual Buffer EncryptVector(const std::vector<double> &cleartext) = 0;

  /*!
   * \brief Decrypt a serialized ciphertext into an array of numbers
   * \param ciphertext A serialzied buffer of ciphertext
   * \return An array of numbers
  */
  virtual std::vector<double> DecryptVector(const std::vector<Buffer> &ciphertext) = 0;

  /*!
   * \brief Add the G&H pairs for a series of samples
   *
   * \param result A vector of `Buffer` objects that will store the aggregated G&H pairs for the samples
   * \param ridx A pointer to an array of row IDs, where each row ID corresponds to a sample
   * \param size The size of the `ridx` array, representing the total number of samples (row IDs) to process
   */
  virtual void AddGHPairs(std::vector<Buffer>& result, const std::uint64_t *ridx, const std::size_t size) = 0;

  /*!
   * \brief Free encrypted data buffer
   * \param ciphertext The buffer for encrypted data
   */
  virtual void FreeEncryptedData(Buffer &ciphertext) {
    if (ciphertext.allocated && ciphertext.buffer != nullptr) {
      free(ciphertext.buffer);
      ciphertext.allocated = false;
    }
    ciphertext.buffer = nullptr;
    ciphertext.buf_size = 0;
  };

  /*!
   * \brief Prepare the bin index vector
   *
   * \param binIndexVec A vector of vectors of row IDs, where each inner vector contains row IDs belonging
   *        to the same bin. The outer vector has a length equal to the total number of bins (`total_bin_size`).
   * \param ridx A pointer to an array of row IDs, where each row ID corresponds to a sample
   * \param size The size of the `ridx` array, representing the total number of samples (row IDs) to process
   *
   */
  void prepareBinIndexVec(std::vector<std::vector<int>>& binIndexVec, const std::uint64_t *ridx, const std::size_t size);

private:

  /*!
   * \brief Build histograms in encrypted space for vertical training
   *
   * \param ridx Pointer to a matrix of row IDs for each node
   * \param sizes An array of sizes of each node
   * \param nidx An array for each node ID
   * \param len Number of nodes
   * \param out_hist Pointer to encrypted histogram buffer
   * \param out_len Buffer size
   */
  void BuildEncryptedHistVertActive(const std::uint64_t **ridx, const std::size_t *sizes, const std::int32_t *nidx,
                                    std::size_t len, std::uint8_t **out_hist, std::size_t *out_len);


  /*!
   * \brief Build histograms in encrypted space for vertical training
   *
   * \param ridx Pointer to a matrix of row IDs for each node
   * \param sizes An array of sizes of each node
   * \param nidx An array for each node ID
   * \param len Number of nodes
   * \param out_hist Pointer to encrypted histogram buffer
   * \param out_len Buffer size
   */
  void BuildEncryptedHistVertPassive(const std::uint64_t **ridx, const std::size_t *sizes, const std::int32_t *nidx,
                                     std::size_t len, std::uint8_t **out_hist, std::size_t *out_len);

};

} // namespace nvflare
