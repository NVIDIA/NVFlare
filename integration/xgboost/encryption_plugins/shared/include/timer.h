/*
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

#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    Timer() : start_time_(), end_time_() {
      begin_time_ = std::chrono::high_resolution_clock::now();
    }

    void start() {
      start_time_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
      end_time_ = std::chrono::high_resolution_clock::now();
    }

    double duration() const {
      return std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count();
    }

    double now() {
      return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - begin_time_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point begin_time_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

#endif // TIMER_H
