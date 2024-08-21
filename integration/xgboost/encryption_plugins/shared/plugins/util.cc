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
#include <set>
#include <algorithm>
#include "util.h"


constexpr double kScaleFactor = 1000000.0;

std::vector<std::pair<int, int>> distribute_work(size_t num_jobs, size_t const num_workers) {
    std::vector<std::pair<int, int>> result;
    auto num = num_jobs / num_workers;
    auto remainder = num_jobs % num_workers;
    int start = 0;
    for (int i = 0; i < num_workers; i++) {
        auto stop = static_cast<int>((start + num - 1));
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
    for (auto &item: result) {
        sum += item.second - item.first + 1;
    }

    if (sum != num_jobs) {
        std::cout << "Distribution error" << std::endl;
    }

    return result;
}

uint32_t to_int(double d) {
    auto int_val = static_cast<int32_t>(d * kScaleFactor);
    return static_cast<uint32_t>(int_val);
}

double to_double(uint32_t i) {
    auto int_val = static_cast<int32_t>(i);
    return static_cast<double>(int_val / kScaleFactor);
}

std::string get_string(std::vector<std::pair<std::string_view, std::string_view>> const &args,
                       std::string_view const &key, std::string_view const default_value) {

    auto it = find_if(
            args.begin(), args.end(),
            [key](const auto &p) { return p.first == key; });

    if (it != args.end()) {
        return std::string{it->second};
    }

    return std::string{default_value};
}

bool get_bool(std::vector<std::pair<std::string_view, std::string_view>> const &args,
              const std::string &key, bool default_value) {
    std::string value = get_string(args, key, "");
    if (value.empty()) {
        return default_value;
    }
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    auto true_values = std::set < std::string_view > {"true", "yes", "y", "on", "1"};
    return true_values.count(value) > 0;
}

int get_int(std::vector<std::pair<std::string_view, std::string_view>> const &args,
            const std::string &key, int default_value) {

    auto value = get_string(args, key, "");
    if (value.empty()) {
        return default_value;
    }

    return stoi(value, nullptr);
}
