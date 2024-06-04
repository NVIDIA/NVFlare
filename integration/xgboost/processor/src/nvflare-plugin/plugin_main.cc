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
#include <cstring>
#include "nvflare_processor.h"
#include "local_mock.h"

extern "C" {

processing::Processor *LoadProcessor(char *plugin_name) {
    if (strcasecmp(plugin_name, "nvflare") == 0) {
        return new NVFlareProcessor();
    } if (strcasecmp(plugin_name, "nvflare:mock") == 0) {
        return new LocalMockProcessor();
    } else {
        std::cout << "Unknown plugin name: " << plugin_name << std::endl;
        return nullptr;
    }
}

}  // extern "C"
