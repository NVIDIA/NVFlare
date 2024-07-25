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
#include <memory>      // for shared_ptr
#include <stdexcept>   // for invalid_argument
#include <string_view> // for string_view
#include <vector>      // for vector
#include <algorithm>   // for transform

#include "delegated_plugin.h"

// Opaque pointer type for the C API.
typedef void *FederatedPluginHandle; // NOLINT

namespace nvflare {
namespace {
// The opaque type for the C handle.
using CHandleT = std::shared_ptr<BasePlugin> *;
// Actual representation used in C++ code base.
using HandleT = std::remove_pointer_t<CHandleT>;

std::string &GlobalErrorMsg() {
  static thread_local std::string msg;
  return msg;
}

// Perform handle handling for C API functions.
template <typename Fn> auto CApiGuard(FederatedPluginHandle handle, Fn &&fn) {
  auto pptr = static_cast<CHandleT>(handle);
  if (!pptr) {
    return 1;
  }

  try {
    if constexpr (std::is_void_v<std::invoke_result_t<Fn, decltype(*pptr)>>) {
      fn(*pptr);
      return 0;
    } else {
      return fn(*pptr);
    }
  } catch (std::exception const &e) {
    GlobalErrorMsg() = e.what();
    return 1;
  }
}
} // namespace
} // namespace nvflare

#if defined(_MSC_VER) || defined(_WIN32)
#define NVF_C __declspec(dllexport)
#else
#define NVF_C __attribute__((visibility("default")))
#endif // defined(_MSC_VER) || defined(_WIN32)

extern "C" {
NVF_C char const *FederatedPluginErrorMsg() {
  return nvflare::GlobalErrorMsg().c_str();
}

FederatedPluginHandle NVF_C FederatedPluginCreate(int argc, char const **argv) {
  // std::cout << "==== FedreatedPluginCreate called with argc=" << argc << std::endl;
  using namespace nvflare;
  try {
    auto pptr = new std::shared_ptr<BasePlugin>;
    std::vector<std::pair<std::string_view, std::string_view>> args;
    std::transform(
        argv, argv + argc, std::back_inserter(args), [](char const *carg) {
          // Split a key value pair in contructor argument: `key=value`
          std::string_view arg{carg};
          auto idx = arg.find('=');
          if (idx == std::string_view::npos) {
            // `=` not found
            throw std::invalid_argument{"Invalid argument:" + std::string{arg}};
          }
          auto key = arg.substr(0, idx);
          auto value = arg.substr(idx + 1);
          return std::make_pair(key, value);
        });
    *pptr = std::make_shared<DelegatedPlugin>(args);
    // std::cout << "==== Plugin created: " << pptr << std::endl;
    return pptr;
  } catch (std::exception const &e) {
    // std::cout << "==== Create exception " << e.what() << std::endl;
    GlobalErrorMsg() = e.what();
    return nullptr;
  }
}

int NVF_C FederatedPluginClose(FederatedPluginHandle handle) {
  using namespace nvflare;
  auto pptr = static_cast<CHandleT>(handle);
  if (!pptr) {
    return 1;
  }

  delete pptr;

  return 0;
}

int NVF_C FederatedPluginEncryptGPairs(FederatedPluginHandle handle,
                                       float const *in_gpair, size_t n_in,
                                       uint8_t **out_gpair, size_t *n_out) {
  using namespace nvflare;
  return CApiGuard(handle, [&](HandleT const &plugin) {
    plugin->EncryptGPairs(in_gpair, n_in, out_gpair, n_out);
    return 0;
  });
}

int NVF_C FederatedPluginSyncEncryptedGPairs(FederatedPluginHandle handle,
                                             uint8_t const *in_gpair,
                                             size_t n_bytes,
                                             uint8_t const **out_gpair,
                                             size_t *n_out) {
  using namespace nvflare;
  return CApiGuard(handle, [&](HandleT const &plugin) {
    plugin->SyncEncryptedGPairs(in_gpair, n_bytes, out_gpair, n_out);
  });
}

int NVF_C FederatedPluginResetHistContextVert(FederatedPluginHandle handle,
                                              uint32_t const *cutptrs,
                                              size_t cutptr_len,
                                              int32_t const *bin_idx,
                                              size_t n_idx) {
  using namespace nvflare;
  return CApiGuard(handle, [&](HandleT const &plugin) {
    plugin->ResetHistContext(cutptrs, cutptr_len, bin_idx, n_idx);
  });
}

int NVF_C FederatedPluginBuildEncryptedHistVert(
    FederatedPluginHandle handle, uint64_t const **ridx, size_t const *sizes,
    int32_t const *nidx, size_t len, uint8_t **out_hist, size_t *out_len) {
  using namespace nvflare;
  return CApiGuard(handle, [&](HandleT const &plugin) {
    plugin->BuildEncryptedHistVert(ridx, sizes, nidx, len, out_hist, out_len);
  });
}

int NVF_C FederatedPluginSyncEncryptedHistVert(FederatedPluginHandle handle,
                                               uint8_t *in_hist, size_t len,
                                               double **out_hist,
                                               size_t *out_len) {
  using namespace nvflare;
  return CApiGuard(handle, [&](HandleT const &plugin) {
    plugin->SyncEncryptedHistVert(in_hist, len, out_hist, out_len);
  });
}

int NVF_C FederatedPluginBuildEncryptedHistHori(FederatedPluginHandle handle,
                                                double const *in_hist,
                                                size_t len, uint8_t **out_hist,
                                                size_t *out_len) {
  using namespace nvflare;
  return CApiGuard(handle, [&](HandleT const &plugin) {
    plugin->BuildEncryptedHistHori(in_hist, len, out_hist, out_len);
  });
}

int NVF_C FederatedPluginSyncEncryptedHistHori(FederatedPluginHandle handle,
                                               uint8_t const *in_hist,
                                               size_t len, double **out_hist,
                                               size_t *out_len) {
  using namespace nvflare;
  return CApiGuard(handle, [&](HandleT const &plugin) {
    plugin->SyncEncryptedHistHori(in_hist, len, out_hist, out_len);
    return 0;
  });
}
} // extern "C"
