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
#include <algorithm>

#include "pass_thru_plugin.h"
#include "data_set_ids.h"

namespace nvflare {

void PassThruPlugin::BuildEncryptedHistHori(const double *in_histogram, std::size_t len,
                                            std::uint8_t **out_hist, std::size_t *out_len) {
  if (debug_) {
    std::cout << Ident() << " PassThruPlugin::BuildEncryptedHistHori called with " << len << " entries" << std::endl;
  }

  DamEncoder encoder(kDataSetHistogramResult, true, dam_debug_);
  auto array = std::vector<double>(in_histogram, in_histogram + len);
  encoder.AddFloatArray(array);
  std::size_t size;
  auto buffer =  encoder.Finish(size);
  buffer_.resize(size);
  std::copy_n(buffer, size, buffer_.begin());
  free(buffer);
  *out_hist = buffer_.data();
  *out_len = buffer_.size();
}

void PassThruPlugin::SyncEncryptedHistHori(const std::uint8_t *buffer, std::size_t len,
                                           double **out_hist, std::size_t *out_len) {
  if (debug_) {
    std::cout << Ident() << " PassThruPlugin::SyncEncryptedHistHori called with buffer size: " << len << std::endl;
  }

  auto remaining = len;
  auto pointer = buffer;

  // The buffer is concatenated by AllGather. It may contain multiple DAM buffers
  std::vector<double>& result = histo_;
  result.clear();
  while (remaining > kPrefixLen) {
    DamDecoder decoder(const_cast<std::uint8_t *>(pointer), remaining, true, dam_debug_);
    if (!decoder.IsValid()) {
      std::cout << "Not DAM encoded histogram ignored at offset: "
                << static_cast<int>(pointer - buffer) << std::endl;
      break;
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

Buffer PassThruPlugin::EncryptVector(const std::vector<double>& cleartext) {
  if (debug_ && cleartext.size() > 2) {
    std::cout << "PassThruPlugin::EncryptVector called with cleartext size: " << cleartext.size() << std::endl;
  }

  size_t size = cleartext.size() * sizeof(double);
  auto buf = static_cast<std::uint8_t *>(malloc(size));
  std::copy_n(reinterpret_cast<std::uint8_t const*>(cleartext.data()), size, buf);

  return {buf, size, true};
}

std::vector<double> PassThruPlugin::DecryptVector(const std::vector<Buffer>& ciphertext) {
  if (debug_) {
    std::cout << "PassThruPlugin::DecryptVector with ciphertext size: " << ciphertext.size() << std::endl;
  }

  std::vector<double> result;

  for (auto const &v : ciphertext) {
    size_t n = v.buf_size/sizeof(double);
    auto p = static_cast<double *>(v.buffer);
    for (int i = 0; i < n; i++) {
      result.push_back(p[i]);
    }
  }

  return result;
}

void PassThruPlugin::AddGHPairs(std::vector<Buffer>& result, const std::uint64_t *ridx, const std::size_t size) {
  size_t total_bin_size = cuts_.back();
  if (debug_) {
    std::cout << "PassThruPlugin::AddGHPairs called with " << total_bin_size << " bins" << std::endl;
  }

  // Can't do this in real plugin. It needs to be broken into encrypted parts
  auto gh_pairs = DecryptVector(std::vector<Buffer>{Buffer(encrypted_gh_.data(), encrypted_gh_.size())});

  std::vector<std::vector<int>> binIndexVec;
  prepareBinIndexVec(binIndexVec, ridx, size);

  size_t total_sample_ids = 0;
  for (auto i = 0; i < binIndexVec.size(); ++i) {
    auto rows = binIndexVec[i];
    total_sample_ids += rows.size();
    double g = 0.0;
    double h = 0.0;

    for (auto row : rows) {
      g += gh_pairs[2 * row];
      h += gh_pairs[2 * row + 1];
    }

    // In real plugin, the sum should be still in encrypted state. No need to do this step
    auto encrypted_sum = EncryptVector(std::vector<double>{g, h});
    // print_buffer(reinterpret_cast<uint8_t *>(encrypted_sum.buffer), encrypted_sum.buf_size);
    result[i] = encrypted_sum;
  }

  if (debug_) {
    std::cout << "PassThruPlugin::AddGHPairs finished with " << total_bin_size << " bins and " << total_sample_ids << " ids " << std::endl;
  }

}

} // namespace nvflare
