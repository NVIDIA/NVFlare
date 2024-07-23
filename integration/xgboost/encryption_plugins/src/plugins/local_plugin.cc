/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#include <iostream>
#include <chrono>
#include <algorithm>
#include "local_plugin.h"
#include "data_set_ids.h"

namespace nvflare {

void LocalPlugin::EncryptGPairs(const float *in_gpair, std::size_t n_in, std::uint8_t **out_gpair, std::size_t *n_out) {
  if (debug_) {
    std::cout << "LocalPlugin::EncryptGPairs called with pairs size: " << n_in << std::endl;
  }

  if (print_timing_) {
    std::cout << "Encrypting " << n_in / 2 << " GH Pairs" << std::endl;
  }
  auto start = std::chrono::system_clock::now();

  auto pairs = std::vector<float>(in_gpair, in_gpair + n_in);
  auto double_pairs = std::vector<double>(pairs.cbegin(), pairs.cend());
  auto encrypted_data = EncryptVector(double_pairs);

  if (print_timing_) {
    auto end = std::chrono::system_clock::now();
    auto secs = (double) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    std::cout << "Encryption time: " << secs << " seconds" << std::endl;
  }

  // Serialize with DAM so the buffers can be separated after all-gather
  DamEncoder encoder(kDataSetGHPairs, true, dam_debug_);
  encoder.AddBuffer(encrypted_data);

  std::size_t size;
  auto buffer = encoder.Finish(size);
  FreeEncryptedData(encrypted_data);
  buffer_.resize(size);
  std::copy_n(buffer, size, buffer_.begin());
  free(buffer);

  *out_gpair = buffer_.data();
  *n_out = buffer_.size();
  if (debug_) {
    std::cout << "Encrypted GPairs:" << std::endl;
    print_buffer(*out_gpair, *n_out);
  }

  // Save pairs for future operations. This is only called on active site
  gh_pairs_ = std::vector<double>(double_pairs);
}

void LocalPlugin::SyncEncryptedGPairs(const std::uint8_t *in_gpair, std::size_t n_bytes,
                                      const std::uint8_t **out_gpair, std::size_t *out_n_bytes) {
  if (debug_) {
    std::cout << "LocalPlugin::SyncEncryptedGPairs called with buffer:" << std::endl;
    print_buffer(const_cast<uint8_t *>(in_gpair), n_bytes);
  }

  *out_n_bytes = n_bytes;
  *out_gpair = in_gpair;
  auto decoder = DamDecoder(const_cast<std::uint8_t *>(in_gpair), n_bytes, true, dam_debug_);
  if (!decoder.IsValid()) {
    std::cout << "LocalPlugin::SyncEncryptedGPairs called with wrong data" << std::endl;
    return;
  }

  auto encrypted_buffer = decoder.DecodeBuffer();
  if (debug_) {
    std::cout << "Encrypted buffer size: " << encrypted_buffer.buf_size << std::endl;
  }

  // The caller may free buffer so a copy is needed
  auto pointer = static_cast<u_int8_t *>(encrypted_buffer.buffer);
  encrypted_gh_ = std::vector<std::uint8_t>(pointer, pointer + encrypted_buffer.buf_size);
  FreeEncryptedData(encrypted_buffer);
}

void LocalPlugin::ResetHistContext(const std::uint32_t *cutptrs, std::size_t cutptr_len, const std::int32_t *bin_idx,
                                   std::size_t n_idx) {
  if (debug_) {
    std::cout << "LocalPlugin::ResetHistContext called with cutptrs size: " << cutptr_len << " bin_idx size: "
              << n_idx << std::endl;
  }

  cuts_ = std::vector<uint32_t>(cutptrs, cutptrs + cutptr_len);
  slots_ = std::vector<int32_t>(bin_idx, bin_idx + n_idx);
}

void LocalPlugin::BuildEncryptedHistHori(const double *in_histogram, std::size_t len, std::uint8_t **out_hist,
                                         std::size_t *out_len) {
  if (debug_) {
    std::cout << "LocalPlugin::BuildEncryptedHistHori called with " << len << " entries" << std::endl;
  }

  // don't have a local implementation yet, just encoded it and let NVFlare handle it.
  DamEncoder encoder(kDataSetHistograms, false, dam_debug_);
  auto histograms = std::vector<double>(in_histogram, in_histogram + len);
  encoder.AddFloatArray(histograms);
  std::size_t size;
  auto buffer = encoder.Finish(size);
  buffer_.resize(size);
  std::copy_n(buffer, size, buffer_.begin());
  free(buffer);

  *out_hist = buffer_.data();
  *out_len = buffer_.size();
}

void LocalPlugin::SyncEncryptedHistHori(const std::uint8_t *buffer, std::size_t len, double **out_hist,
                                        std::size_t *out_len) {
  if (debug_) {
    std::cout << "LocalPlugin::SyncEncryptedHistHori called with buffer size: " << len << std::endl;
  }

  // No local implementation yet, just decode data from NVFlare
  *out_hist = nullptr;
  *out_len = 0;
  DamDecoder decoder(const_cast<std::uint8_t *>(buffer), len, false, dam_debug_);
  if (!decoder.IsValid()) {
    std::cout << "Not DAM encoded buffer, ignored" << std::endl;
    return;
  }

  if (decoder.GetDataSetId() != kDataSetHistogramResult) {
    std::cout << "Invalid dataset for SyncEncryptedHistHori: " << decoder.GetDataSetId() << std::endl;
    return;
  }

  histo_ = decoder.DecodeFloatArray();
  *out_hist = histo_.data();
  *out_len = histo_.size();
}

void LocalPlugin::BuildEncryptedHistVert(const std::uint64_t **ridx, const std::size_t *sizes, const std::int32_t *nidx,
                                         std::size_t len, std::uint8_t **out_hist, std::size_t *out_len) {
  if (debug_) {
    std::cout << "LocalPlugin::BuildEncryptedHistVert called with number of nodes: " << len << std::endl;
  }

  if (gh_pairs_.empty()) {
    BuildEncryptedHistVertPassive(ridx, sizes, nidx, len, out_hist, out_len);
  } else {
    BuildEncryptedHistVertActive(ridx, sizes, nidx, len, out_hist, out_len);
  }
}

void LocalPlugin::BuildEncryptedHistVertActive(const std::uint64_t **ridx, const std::size_t *sizes, const std::int32_t *nidx,
                                               std::size_t len, std::uint8_t **out_hist, std::size_t *out_len) {

  if (debug_) {
    std::cout << "LocalPlugin::BuildEncryptedHistVertActive called with " << len << " nodes" << std::endl;
  }

  auto total_bin_size = cuts_.back();
  auto histo_size = total_bin_size * 2;
  auto total_size = histo_size * len;

  histo_.clear();
  histo_.resize(total_size);
  size_t start = 0;
  for (std::size_t i = 0; i < len; i++) {
    for (std::size_t j = 0; j < sizes[i]; j++) {
      auto row_id = ridx[i][j];
      auto num = cuts_.size() - 1;
      for (std::size_t f = 0; f < num; f++) {
        int slot = slots_[f + num * row_id];
        if ((slot < 0) || (slot >= total_bin_size)) {
          continue;
        }
        auto g = gh_pairs_[row_id * 2];
        auto h = gh_pairs_[row_id * 2 + 1];
        (histo_)[start + slot * 2] += g;
        (histo_)[start + slot * 2 + 1] += h;
      }
    }
    start += histo_size;
  }

  // Histogram is in clear, can't send to all_gather. Just return empty DAM buffer
  auto encoder = DamEncoder(kDataSetAggregationResult, true, dam_debug_);
  encoder.AddBuffer(Buffer());
  std::size_t size;
  auto buffer = encoder.Finish(size);
  buffer_.resize(size);
  std::copy_n(buffer, size, buffer_.begin());
  free(buffer);
  *out_hist = buffer_.data();
  *out_len = size;
}

void LocalPlugin::BuildEncryptedHistVertPassive(const std::uint64_t **ridx, const std::size_t *sizes, const std::int32_t *nidx,
                                                std::size_t len, std::uint8_t **out_hist, std::size_t *out_len) {
  if (debug_) {
    std::cout << "LocalPlugin::BuildEncryptedHistVertPassive called with " << len << " nodes" << std::endl;
  }

  auto num_slot = cuts_.back();
  auto total_size = num_slot * len;

  auto encrypted_histo = std::vector<Buffer>(total_size);
  size_t offset = 0;
  for (std::size_t i = 0; i < len; i++) {
    auto num = cuts_.size() - 1;
    auto row_id_map = std::map<int, std::vector<int>>();

    // Empty slot leaks data so fill everything with empty vectors
    for (int slot = 0; slot < num_slot; slot++) {
      row_id_map.insert({slot, std::vector<int>()});
    }

    for (std::size_t f = 0; f < num; f++) {
      for (std::size_t j = 0; j < sizes[i]; j++) {
        auto row_id = ridx[i][j];
        int slot = slots_[f + num * row_id];
        if ((slot < 0) || (slot >= num_slot)) {
          continue;
        }
        auto &row_ids = row_id_map[slot];
        row_ids.push_back(static_cast<int>(row_id));
      }
    }

    if (print_timing_) {
      std::size_t add_ops = 0;
      for (auto &item: row_id_map) {
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
  std::size_t size;
  auto buffer = encoder.Finish(size);
  for (auto &item: encrypted_histo) {
    FreeEncryptedData(item);
  }
  buffer_.resize(size);
  std::copy_n(buffer, size, buffer_.begin());
  free(buffer);
  *out_hist = buffer_.data();
  *out_len = size;
}

void LocalPlugin::SyncEncryptedHistVert(std::uint8_t *hist_buffer, std::size_t len,
                                        double **out, std::size_t *out_len) {
  if (debug_) {
    std::cout << "LocalPlugin::SyncEncryptedHistVert called with buffer size: " << len << " nodes" << std::endl;
  }

  auto remaining = len;
  auto pointer = hist_buffer;

  *out = nullptr;
  *out_len = 0;
  if (gh_pairs_.empty()) {
    // Do nothing for passive worker
    return;
  }

  // The buffer is concatenated by AllGather. It may contain multiple DAM buffers
  auto first = true;
  auto orig_size = histo_.size();
  while (remaining > kPrefixLen) {
    DamDecoder decoder(pointer, remaining, true, dam_debug_);
    if (!decoder.IsValid()) {
      std::cout << "Not DAM encoded buffer ignored at offset: "
                << static_cast<int>((pointer - hist_buffer)) << std::endl;
      break;
    }
    auto size = decoder.Size();
    if (first) {
      if (histo_.empty()) {
        std::cout << "No clear histogram." << std::endl;
        return;
      }
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

      if (decrypted_histo.size() != orig_size) {
        std::cout << "Histo sizes are different: " << decrypted_histo.size()
                  << " != " << orig_size << std::endl;
      }
      histo_.insert(histo_.end(), decrypted_histo.cbegin(), decrypted_histo.cend());
    }
    remaining -= size;
    pointer += size;
  }

  if (debug_) {
    std::cout << "Decrypted result size: " << histo_.size() << std::endl;
  }

  // print_buffer(reinterpret_cast<uint8_t *>(result.data()), result.size()*8);

  *out = histo_.data();
  *out_len = histo_.size();
}

} // namespace nvflare
