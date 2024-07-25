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
#include "delegated_plugin.h"
#include "pass_thru_plugin.h"
#include "nvflare_plugin.h"

namespace nvflare {

DelegatedPlugin::DelegatedPlugin(std::vector<std::pair<std::string_view, std::string_view>> const &args):
  BasePlugin(args) {

  auto name = get_string(args, "name");
  // std::cout << "==== Name is " << name << std::endl;
  if (name == "pass-thru") {
    plugin_ = new PassThruPlugin(args);
  } else if (name == "nvflare") {
    plugin_ = new NvflarePlugin(args);
  } else {
    throw std::invalid_argument{"Unknown plugin name: " + name};
  }
}

} // namespace nvflare
