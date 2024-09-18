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
#include <vector>
#include <map>

constexpr char kSignature[] = "NVDADAM1";       // DAM (Direct Accessible Marshalling) V1
constexpr char kSignatureLocal[] = "NVDADAML";  // DAM Local version
constexpr int kPrefixLen = 24;

constexpr int kDataTypeInt = 1;
constexpr int kDataTypeFloat = 2;
constexpr int kDataTypeString = 3;
constexpr int kDataTypeBuffer = 4;
constexpr int kDataTypeIntArray = 257;
constexpr int kDataTypeFloatArray = 258;
constexpr int kDataTypeBufferArray = 259;
constexpr int kDataTypeMap = 1025;

/*! \brief A replacement for std::span */
class Buffer {
public:
    void *buffer;
    size_t buf_size;
    bool allocated;

    Buffer() : buffer(nullptr), buf_size(0), allocated(false) {
    }

    Buffer(void *buffer, size_t buf_size, bool allocated=false) :
        buffer(buffer), buf_size(buf_size), allocated(allocated) {
    }

    Buffer(const Buffer &that):
        buffer(that.buffer), buf_size(that.buf_size), allocated(false) {
    }
};

class Entry {
 public:
    int64_t data_type;
    const uint8_t * pointer;
    int64_t size;

    Entry(int64_t data_type, const uint8_t *pointer, int64_t size) {
        this->data_type = data_type;
        this->pointer = pointer;
        this->size = size;
    }

    [[nodiscard]] std::size_t ItemSize() const
    {
        size_t item_size;
        switch (data_type) {
            case kDataTypeBuffer:
            case kDataTypeString:
            case kDataTypeBufferArray:
                item_size = 1;
                break;
            default:
                item_size = 8;
        }
        return item_size;
    }
};

class DamEncoder {
 private:
    bool encoded_ = false;
    bool local_version_ = false;
    bool debug_ = false;
    int64_t data_set_id_;
    std::vector<Entry> entries_;

 public:
    explicit DamEncoder(int64_t data_set_id, bool local_version=false, bool debug=false) {
        data_set_id_ = data_set_id;
        local_version_ = local_version;
        debug_ = debug;

    }

    void AddBuffer(const Buffer &buffer);

    void AddIntArray(const std::vector<int64_t> &value);

    void AddFloatArray(const std::vector<double> &value);

    void AddBufferArray(const std::vector<Buffer> &value);

    std::uint8_t * Finish(size_t &size);

 private:
    std::size_t CalculateSize();
};

class DamDecoder {
 private:
    bool local_version_ = false;
    std::uint8_t *buffer_ = nullptr;
    std::size_t buf_size_ = 0;
    std::uint8_t *pos_ = nullptr;
    std::size_t remaining_ = 0;
    int64_t data_set_id_ = 0;
    int64_t len_ = 0;
    bool debug_ = false;

 public:
    explicit DamDecoder(std::uint8_t *buffer, std::size_t size, bool local_version=false, bool debug=false);

    [[nodiscard]] std::size_t Size() const {
        return len_;
    }

    [[nodiscard]] int64_t GetDataSetId() const {
        return data_set_id_;
    }

    [[nodiscard]] bool IsValid() const;

    Buffer DecodeBuffer();

    std::vector<int64_t> DecodeIntArray();

    std::vector<double> DecodeFloatArray();

    std::vector<Buffer> DecodeBufferArray();
};

void print_buffer(const uint8_t *buffer, std::size_t size);
