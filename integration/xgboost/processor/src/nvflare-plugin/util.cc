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
#include "util.h"
#include <iostream>
#include <set>
#include <algorithm>

const double kScaleFactor = 1000000.0;

std::vector<std::pair<int, int>> distribute_work(size_t num_jobs, size_t num_workers) {
    std::vector<std::pair<int, int>> result;
    auto num = num_jobs/num_workers;
    auto remainder = num_jobs%num_workers;
    int start = 0;
    for (int i = 0; i < num_workers; i++) {
        auto stop = (int)(start + num - 1);
        if (i < remainder) {
            // If jobs cannot be evenly distributed, first few workers take an extra one
            stop += 1;
        }

        if (start <= stop) {
            result.emplace_back(start, stop);
        }
        start = stop + 1;
    }

    // Verify all jobs are distributed
    int sum = 0;
    for (auto &item : result) {
        sum += item.second - item.first + 1;
    }

    if (sum != num_jobs) {
        std::cout << "Distribution error" << std::endl;
    }

    return result;
}

uint32_t to_int(double d) {
    auto int_val = (int32_t)(d*kScaleFactor);
    return (uint32_t)int_val;
}

double to_double(uint32_t i) {
    auto int_val = (int32_t)i;
    return (double)(int_val/kScaleFactor);
}

std::string get_string(const std::map<std::string, std::string>& params, const std::string key,
                       std::string default_value) {
    auto it = params.find(key);
    if (it == params.end()) {
        return default_value;
    }

    return it->second;
}

bool get_bool(const std::map<std::string, std::string>& params, const std::string key, bool default_value) {
    auto value = get_string(params, key, "");
    if (value.empty()) {
        return default_value;
    }
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return std::tolower(c);});
    auto true_values = std::set<std::string>{"true", "yes", "y", "on", "1"};
    return true_values.count(value) > 0;
}

int get_int(const std::map<std::string, std::string>& params, const std::string key, int default_value) {

    auto value = get_string(params, key, "");
    if (value == "") {
        return default_value;
    }

    return stoi(value, nullptr);
}
