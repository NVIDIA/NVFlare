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
#include <string>
#include <vector>
#include <map>

const char kSignature[] = "NVDADAM1";  // DAM (Direct Accessible Marshalling) V1
const int kPrefixLen = 24;

const int kDataTypeInt = 1;
const int kDataTypeFloat = 2;
const int kDataTypeString = 3;
const int kDataTypeIntArray = 257;
const int kDataTypeFloatArray = 258;

const int kDataTypeMap = 1025;

class Entry {
 public:
    int64_t data_type;
    uint8_t * pointer;
    int64_t size;

    Entry(int64_t data_type, uint8_t *pointer, int64_t size) {
        this->data_type = data_type;
        this->pointer = pointer;
        this->size = size;
    }
};

class DamEncoder {
 private:
    bool encoded = false;
    int64_t data_set_id;
    std::vector<Entry *> *entries = new std::vector<Entry *>();

 public:
    explicit DamEncoder(int64_t data_set_id) {
        this->data_set_id = data_set_id;
    }

    void AddIntArray(const std::vector<int64_t> &value);

    void AddFloatArray(const std::vector<double> &value);

    std::uint8_t * Finish(size_t &size);

 private:
    std::size_t calculate_size();
};

class DamDecoder {
 private:
    std::uint8_t *buffer = nullptr;
    std::size_t buf_size = 0;
    std::uint8_t *pos = nullptr;
    std::size_t remaining = 0;
    int64_t data_set_id = 0;
    int64_t len = 0;

 public:
    explicit DamDecoder(std::uint8_t *buffer, std::size_t size);

    size_t Size() {
        return len;
    }

    int64_t GetDataSetId() {
        return data_set_id;
    }

    bool IsValid();

    std::vector<int64_t> DecodeIntArray();

    std::vector<double> DecodeFloatArray();
};

void print_buffer(uint8_t *buffer, int size);
