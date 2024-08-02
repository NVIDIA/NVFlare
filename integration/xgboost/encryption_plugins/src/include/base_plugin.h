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
#include <sstream>
#include <iomanip>
#include <unistd.h>

#include "util.h"

namespace nvflare {

/**
 * @brief Abstract interface for the encryption plugin
 *
 * All plugin implementations must inherit this class.
 */
class BasePlugin {
protected:
  bool debug_ = false;
  bool print_timing_ = false;
  bool dam_debug_ = false;

public:
/**
 * @brief Constructor
 *
 * All inherited classes should call this constructor.
 *
 * @param args Entries from federated_plugin in communicator environments.
 */
  explicit BasePlugin(
      std::vector<std::pair<std::string_view, std::string_view>> const &args) {
    debug_ = get_bool(args, "debug");
    print_timing_ = get_bool(args, "print_timing");
    dam_debug_ = get_bool(args, "dam_debug");
  }

  /**
   * @brief Destructor
   */
  virtual ~BasePlugin() = default;

  /**
   * @brief Identity for the plugin used for debug
   *
   * This is a string with instance address and process id.
   */
  std::string Ident() {
    std::stringstream ss;
    ss << std::hex << std::uppercase << std::setw(sizeof(void*) * 2) << std::setfill('0') <<
      reinterpret_cast<uintptr_t>(this);
    return ss.str() + "-" + std::to_string(getpid());
  }

  /**
   * @brief Encrypt the gradient pairs
   *
   * @param in_gpair Input g and h pairs for each record
   * @param n_in The array size (2xnum_of_records)
   * @param out_gpair Pointer to encrypted buffer
   * @param n_out Encrypted buffer size
   */
  virtual void EncryptGPairs(float const *in_gpair, std::size_t n_in,
                             std::uint8_t **out_gpair, std::size_t *n_out) = 0;

  /**
   * @brief Process encrypted gradient pairs
   *
   * @param in_gpair Encrypted gradient pairs
   * @param n_bytes Buffer size of Encrypted gradient
   * @param out_gpair Pointer to decrypted gradient pairs
   * @param out_n_bytes Decrypted buffer size
   */
  virtual void SyncEncryptedGPairs(std::uint8_t const *in_gpair, std::size_t n_bytes,
                                   std::uint8_t const **out_gpair,
                                   std::size_t *out_n_bytes) = 0;

  /**
   * @brief Reset the histogram context
   *
   * @param cutptrs Cut-pointers for the flattened histograms
   * @param cutptr_len cutptrs array size (number of features plus one)
   * @param bin_idx An array (flattened matrix) of slot index for each record/feature
   * @param n_idx The size of above array
   */
  virtual void ResetHistContext(std::uint32_t const *cutptrs, std::size_t cutptr_len,
                                std::int32_t const *bin_idx, std::size_t n_idx) = 0;

  /**
   * @brief Encrypt histograms for horizontal training
   *
   * @param in_histogram The array for the histogram
   * @param len The array size
   * @param out_hist Pointer to encrypted buffer
   * @param out_len Encrypted buffer size
   */
  virtual void BuildEncryptedHistHori(double const *in_histogram, std::size_t len,
                                      std::uint8_t **out_hist, std::size_t *out_len) = 0;

  /**
   * @brief Process encrypted histograms for horizontal training
   *
   * @param buffer Buffer for encrypted histograms
   * @param len Buffer size of encrypted histograms
   * @param out_hist Pointer to decrypted histograms
   * @param out_len Size of above array
   */
  virtual void SyncEncryptedHistHori(std::uint8_t const *buffer, std::size_t len,
                                     double **out_hist, std::size_t *out_len) = 0;

  /**
   * @brief Build histograms in encrypted space for vertical training
   *
   * @param ridx Pointer to a matrix of row IDs for each node
   * @param sizes An array of sizes of each node
   * @param nidx An array for each node ID
   * @param len Number of nodes
   * @param out_hist Pointer to encrypted histogram buffer
   * @param out_len Buffer size
   */
  virtual void BuildEncryptedHistVert(std::uint64_t const **ridx,
                                      std::size_t const *sizes,
                                      std::int32_t const *nidx, std::size_t len,
                                      std::uint8_t **out_hist, std::size_t *out_len) = 0;

  /**
   * @brief Decrypt histogram for vertical training
   *
   * @param hist_buffer Encrypted histogram buffer
   * @param len Buffer size of encrypted histogram
   * @param out Pointer to decrypted histograms
   * @param out_len Size of above array
   */
  virtual void SyncEncryptedHistVert(std::uint8_t *hist_buffer, std::size_t len,
                                     double **out, std::size_t *out_len) = 0;
};
} // namespace nvflare
