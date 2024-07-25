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
#include <cstring>
#include "dam.h"


void print_hex(const uint8_t *buffer, std::size_t size) {
    std::cout << std::hex;
    for (int i = 0; i < size; i++) {
        int c = buffer[i];
        std::cout << c << " ";
    }
    std::cout << std::endl << std::dec;
}

void print_buffer(const uint8_t *buffer, std::size_t size) {
    if (size <= 64) {
        std::cout << "Whole buffer: " << size << " bytes" << std::endl;
        print_hex(buffer, size);
        return;
    }

    std::cout << "First chunk, Total: " << size << " bytes" << std::endl;
    print_hex(buffer, 32);
    std::cout << "Last chunk, Offset: " << size-16 << " bytes" << std::endl;
    print_hex(buffer+size-32, 32);
}

size_t align(const size_t length) {
    return ((length + 7)/8)*8;
}

// DamEncoder ======
void  DamEncoder::AddBuffer(const Buffer &buffer) {
    if (debug_) {
        std::cout << "AddBuffer called, size:  " << buffer.buf_size << std::endl;
    }
    if (encoded_) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    // print_buffer(buffer, buf_size);
    entries_.emplace_back(kDataTypeBuffer, static_cast<const uint8_t *>(buffer.buffer), buffer.buf_size);
}

void DamEncoder::AddFloatArray(const std::vector<double> &value) {
    if (debug_) {
        std::cout << "AddFloatArray called, size:  " << value.size() << std::endl;
    }

    if (encoded_) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    // print_buffer(reinterpret_cast<uint8_t *>(value.data()), value.size() * 8);
    entries_.emplace_back(kDataTypeFloatArray, reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

void  DamEncoder::AddIntArray(const std::vector<int64_t> &value) {
    if (debug_) {
        std::cout << "AddIntArray called, size:  " << value.size() << std::endl;
    }

    if (encoded_) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    // print_buffer(buffer, buf_size);
    entries_.emplace_back(kDataTypeIntArray, reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

void  DamEncoder::AddBufferArray(const std::vector<Buffer> &value) {
    if (debug_) {
        std::cout << "AddBufferArray called, size:  " << value.size() << std::endl;
    }

    if (encoded_) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    size_t size = 0;
    for (auto &buf: value) {
        size += buf.buf_size;
    }
    size += 8*value.size();
    entries_.emplace_back(kDataTypeBufferArray, reinterpret_cast<const uint8_t *>(&value), size);
}


std::uint8_t * DamEncoder::Finish(size_t &size) {
    encoded_ = true;

    size = CalculateSize();
    auto buf = static_cast<uint8_t *>(calloc(size, 1));
    auto pointer = buf;
    auto sig = local_version_ ? kSignatureLocal : kSignature;
    memcpy(pointer, sig, strlen(sig));
    memcpy(pointer+8, &size, 8);
    memcpy(pointer+16, &data_set_id_, 8);

    pointer += kPrefixLen;
    for (auto& entry : entries_) {
        std::size_t len;
        if (entry.data_type == kDataTypeBufferArray) {
            auto buffers = reinterpret_cast<const std::vector<Buffer> *>(entry.pointer);
            memcpy(pointer, &entry.data_type, 8);
            pointer += 8;
            auto array_size = static_cast<int64_t>(buffers->size());
            memcpy(pointer, &array_size, 8);
            pointer += 8;
            auto sizes = reinterpret_cast<int64_t *>(pointer);
            for (auto &item : *buffers) {
                *sizes = static_cast<int64_t>(item.buf_size);
                sizes++;
            }
            len = 8*buffers->size();
            auto buf_ptr = pointer + len;
            for (auto &item : *buffers) {
                if (item.buf_size > 0) {
                    memcpy(buf_ptr, item.buffer, item.buf_size);
                }
                buf_ptr += item.buf_size;
                len += item.buf_size;
            }
        } else {
            memcpy(pointer, &entry.data_type, 8);
            pointer += 8;
            memcpy(pointer, &entry.size, 8);
            pointer += 8;
            len = entry.size * entry.ItemSize();
            if (len) {
                memcpy(pointer, entry.pointer, len);
            }
        }
        pointer += align(len);
    }

    if ((pointer - buf) != size) {
        std::cout << "Invalid encoded size: " << (pointer - buf) << std::endl;
        return nullptr;
    }

    return buf;
}

std::size_t DamEncoder::CalculateSize() {
    std::size_t size = kPrefixLen;

    for (auto& entry : entries_) {
        size += 16;  // The Type and Len
        auto len = entry.size * entry.ItemSize();
        size += align(len);
    }

    return size;
}


// DamDecoder ======

DamDecoder::DamDecoder(std::uint8_t *buffer, std::size_t size, bool local_version, bool debug) {
    local_version_ = local_version;
    buffer_ = buffer;
    buf_size_ = size;
    pos_ = buffer + kPrefixLen;
    debug_ = debug;

    if (size >= kPrefixLen) {
        memcpy(&len_, buffer + 8, 8);
        memcpy(&data_set_id_, buffer + 16, 8);
    } else {
        len_ = 0;
        data_set_id_ = 0;
    }
}

bool DamDecoder::IsValid() const {
    auto sig = local_version_ ? kSignatureLocal : kSignature;
    return buf_size_ >= kPrefixLen && memcmp(buffer_, sig, strlen(sig)) == 0;
}

Buffer DamDecoder::DecodeBuffer() {
    auto type = *reinterpret_cast<int64_t *>(pos_);
    if (type != kDataTypeBuffer) {
        std::cout << "Data type " << type << " doesn't match bytes" << std::endl;
        return {};
    }
    pos_ += 8;

    auto size = *reinterpret_cast<int64_t *>(pos_);
    pos_ += 8;

    if (size == 0) {
        return {};
    }

    auto ptr = reinterpret_cast<void *>(pos_);
    pos_ += align(size);
    return{ ptr, static_cast<std::size_t>(size)};
}

std::vector<int64_t> DamDecoder::DecodeIntArray() {
    auto type = *reinterpret_cast<int64_t *>(pos_);
    if (type != kDataTypeIntArray) {
        std::cout << "Data type " << type << " doesn't match Int Array" << std::endl;
        return {};
    }
    pos_ += 8;

    auto array_size = *reinterpret_cast<int64_t *>(pos_);
    pos_ += 8;
    auto ptr = reinterpret_cast<int64_t *>(pos_);
    pos_ += align(8 * array_size);
    return {ptr, ptr + array_size};
}

std::vector<double> DamDecoder::DecodeFloatArray() {
    auto type = *reinterpret_cast<int64_t *>(pos_);
    if (type != kDataTypeFloatArray) {
        std::cout << "Data type " << type << " doesn't match Float Array" << std::endl;
        return {};
    }
    pos_ += 8;

    auto array_size = *reinterpret_cast<int64_t *>(pos_);
    pos_ += 8;

    auto ptr = reinterpret_cast<double *>(pos_);
    pos_ += align(8 * array_size);
    return {ptr, ptr + array_size};
}

std::vector<Buffer> DamDecoder::DecodeBufferArray() {
    auto type = *reinterpret_cast<int64_t *>(pos_);
    if (type != kDataTypeBufferArray) {
        std::cout << "Data type " << type << " doesn't match Bytes Array" << std::endl;
        return {};
    }
    pos_ += 8;

    auto num = *reinterpret_cast<int64_t *>(pos_);
    pos_ += 8;

    auto size_ptr = reinterpret_cast<int64_t *>(pos_);
    auto buf_ptr = pos_ + 8 * num;
    size_t total_size = 8 * num;
    auto result = std::vector<Buffer>(num);
    for (int i = 0; i < num; i++) {
        auto size = size_ptr[i];
        if (buf_size_ > 0) {
            result[i].buf_size = size;
            result[i].buffer = buf_ptr;
            buf_ptr += size;
        }
        total_size += size;
    }

    pos_ += align(total_size);
    return result;
}
