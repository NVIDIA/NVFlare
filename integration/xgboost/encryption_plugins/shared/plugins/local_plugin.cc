/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#include <iostream>
#include <algorithm>
#include <chrono>
#include "local_plugin.h"
#include "data_set_ids.h"

namespace nvflare {

void LocalPlugin::EncryptGPairs(const float *in_gpair, std::size_t n_in, std::uint8_t **out_gpair, std::size_t *n_out) {
  if (debug_) {
    std::cout << Ident() << " LocalPlugin::EncryptGPairs called with pairs size: " << n_in << std::endl;
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
    auto secs = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0;
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
    std::cout << Ident() << " LocalPlugin::SyncEncryptedGPairs called with buffer:" << std::endl;
    print_buffer(in_gpair, n_bytes);
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
    std::cout << Ident() << " LocalPlugin::ResetHistContext called with cutptrs size: " << cutptr_len << " bin_idx size: "
              << n_idx << std::endl;
  }

  cuts_ = std::vector<uint32_t>(cutptrs, cutptrs + cutptr_len);
  bin_idx_vec_ = std::vector<int32_t>(bin_idx, bin_idx + n_idx);
}

void LocalPlugin::BuildEncryptedHistHori(const double *in_histogram, std::size_t len, std::uint8_t **out_hist,
                                         std::size_t *out_len) {
  if (debug_) {
    std::cout << Ident() << " LocalPlugin::BuildEncryptedHistHori called with " << len << " entries" << std::endl;
    print_buffer(reinterpret_cast<const uint8_t*>(in_histogram), len);
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
  if (debug_) {
    std::cout << "Output buffer" << std::endl;
    print_buffer(*out_hist, *out_len);
  }
}

void LocalPlugin::SyncEncryptedHistHori(const std::uint8_t *buffer, std::size_t len, double **out_hist,
                                        std::size_t *out_len) {
  if (debug_) {
    std::cout << Ident() << " LocalPlugin::SyncEncryptedHistHori called with buffer size: " << len << std::endl;
    print_buffer(buffer, len);
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

  if (debug_) {
    std::cout << "Output buffer" << std::endl;
    print_buffer(reinterpret_cast<const uint8_t*>(*out_hist), histo_.size() * sizeof(double));
  }
}

void LocalPlugin::BuildEncryptedHistVert(const std::uint64_t **ridx, const std::size_t *sizes, const std::int32_t *nidx,
                                         std::size_t len, std::uint8_t **out_hist, std::size_t *out_len) {
  if (debug_) {
    std::cout << Ident() << " LocalPlugin::BuildEncryptedHistVert called with number of nodes: " << len << std::endl;
  }

  if (gh_pairs_.empty()) {
    BuildEncryptedHistVertPassive(ridx, sizes, nidx, len, out_hist, out_len);
  } else {
    BuildEncryptedHistVertActive(ridx, sizes, nidx, len, out_hist, out_len);
  }

  if (debug_) {
    std::cout << "Encrypted histogram output:" << std::endl;
    print_buffer(*out_hist, *out_len);
  }
}

void LocalPlugin::BuildEncryptedHistVertActive(const std::uint64_t **ridx, const std::size_t *sizes, const std::int32_t *nidx,
                                               std::size_t len, std::uint8_t **out_hist, std::size_t *out_len) {

  auto total_bin_size = cuts_.back();
  if (debug_) {
    std::cout << Ident() << " LocalPlugin::BuildEncryptedHistVertActive called with " << len << " nodes, total_bin_size " << total_bin_size << std::endl;
  }

  auto histo_size = total_bin_size * 2;
  auto total_size = histo_size * len;

  histo_.clear();
  histo_.resize(total_size);
  size_t start = 0;
  for (std::size_t i = 0; i < len; i++) {
    for (std::size_t j = 0; j < sizes[i]; j++) {
      auto row_id = ridx[i][j];
      auto num_feature = cuts_.size() - 1;

      for (std::size_t f = 0; f < num_feature; f++) {
        int bin_idx = bin_idx_vec_[f + num_feature * row_id];
        if ((bin_idx < 0) || (bin_idx >= total_bin_size)) {
          continue;
        }
        auto g = gh_pairs_[row_id * 2];
        auto h = gh_pairs_[row_id * 2 + 1];
        (histo_)[start + bin_idx * 2] += g;
        (histo_)[start + bin_idx * 2 + 1] += h;
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
  auto total_bin_size = cuts_.back();
  auto total_size = total_bin_size * len;
  if (debug_) {
    std::cout << Ident() << " LocalPlugin::BuildEncryptedHistVertPassive called with " << len << " nodes, total_bin_size " << total_bin_size << std::endl;
  }

  auto encrypted_histo = std::vector<Buffer>(total_size);
  size_t offset = 0;
  for (std::size_t node_id = 0; node_id < len; node_id++) {
    auto start = std::chrono::system_clock::now();
    auto encrypted_sum = std::vector<Buffer>(total_bin_size);

    AddGHPairs(encrypted_sum, ridx[node_id], sizes[node_id]);

    if (print_timing_) {
      auto end = std::chrono::system_clock::now();
      auto secs = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0;
      std::cout << "Aggregation time: " << secs << " seconds" << std::endl;
    }

    // Convert map back to array
    for (int bin_idx = 0; bin_idx < total_bin_size; bin_idx++) {
      encrypted_histo[offset + bin_idx] = encrypted_sum[bin_idx];
    }

    offset += total_bin_size;
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
    std::cout << Ident() << " LocalPlugin::SyncEncryptedHistVert called with buffer size: " << len << " nodes" << std::endl;
    print_buffer(hist_buffer, len);
  }

  auto remaining = len;
  auto pointer = hist_buffer;

  *out = nullptr;
  *out_len = 0;
  if (gh_pairs_.empty()) {
    if (debug_) {
      std::cout << Ident() << " LocalPlugin::SyncEncryptedHistVert Do nothing for passive worker" << std::endl;
    }
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
        auto secs = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0;
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
    std::cout << Ident() << " Decrypted result size: " << histo_.size() << std::endl;
  }

  // print_buffer(reinterpret_cast<uint8_t *>(result.data()), result.size()*8);

  *out = histo_.data();
  *out_len = histo_.size();
}

void LocalPlugin::prepareBinIndexVec(std::vector<std::vector<int>>& binIndexVec, const std::uint64_t *ridx, const std::size_t size) {
  auto total_bin_size = cuts_.back();
  auto num_feature = cuts_.size() - 1;
  std::vector<size_t> bin_counts(total_bin_size, 0);  // Store the count of row_ids per bin

  for (std::size_t f = 0; f < num_feature; f++) {
    for (std::size_t j = 0; j < size; j++) {
      auto row_id = ridx[j];
      int bin_idx = bin_idx_vec_[f + num_feature * row_id];
      if ((bin_idx < 0) || (bin_idx >= total_bin_size)) {
        continue;
      }
      bin_counts[bin_idx]++;
    }
  }

  binIndexVec.resize(total_bin_size);
  for (auto i = 0; i < total_bin_size; i++) {
    binIndexVec[i].resize(bin_counts[i]);
  }

  std::vector<size_t> bin_insert_index(total_bin_size, 0);  // Track the current insertion index for each bin

  // second pass
  for (std::size_t f = 0; f < num_feature; f++) {
    for (std::size_t j = 0; j < size; j++) {
        auto row_id = ridx[j];
        int bin_idx = bin_idx_vec_[f + num_feature * row_id];

        if ((bin_idx < 0) || (bin_idx >= total_bin_size)) {
            continue;  // Skip invalid bin indices
        }

        size_t insert_pos = bin_insert_index[bin_idx];
        binIndexVec[bin_idx][insert_pos] = static_cast<int>(row_id);
        bin_insert_index[bin_idx]++;
    }
  }

}

} // namespace nvflare
