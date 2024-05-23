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

void print_buffer(uint8_t *buffer, int size) {
    for (int i = 0; i < size; i++) {
        auto c = buffer[i];
        std::cout << std::hex << (int) c << " ";
    }
    std::cout << std::endl << std::dec;
}

size_t align(const size_t length) {
    return ((length + 7)/8)*8;
}

// DamEncoder ======
void DamEncoder::AddFloatArray(const std::vector<double> &value) {
    if (encoded) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    // print_buffer(reinterpret_cast<uint8_t *>(value.data()), value.size() * 8);
    entries->push_back(new Entry(kDataTypeFloatArray, reinterpret_cast<const uint8_t *>(value.data()), value.size()));
}

void  DamEncoder::AddIntArray(const std::vector<int64_t> &value) {
    std::cout << "AddIntArray called, size:  " << value.size() << std::endl;
    if (encoded) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    // print_buffer(buffer, buf_size);
    entries->push_back(new Entry(kDataTypeIntArray, reinterpret_cast<const uint8_t *>(value.data()), value.size()));
}

void  DamEncoder::AddBytes(const void *buffer, const size_t buf_size) {
    std::cout << "AddBytes called, size:  " << buf_size << std::endl;
    if (encoded) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    // print_buffer(buffer, buf_size);
    entries->push_back(new Entry(kDataTypeBytes, reinterpret_cast<const uint8_t *>(buffer), buf_size));
}

std::uint8_t * DamEncoder::Finish(size_t &size) {
    encoded = true;

    size = CalculateSize();
    auto buf = static_cast<uint8_t *>(calloc(size, 1));
    auto pointer = buf;
    auto sig = local_version ? kSignatureLocal : kSignature;
    memcpy(pointer, sig, strlen(sig));
    memcpy(pointer+8, &size, 8);
    memcpy(pointer+16, &data_set_id, 8);

    pointer += kPrefixLen;
    for (auto entry : *entries) {
        memcpy(pointer, &entry->data_type, 8);
        pointer += 8;
        memcpy(pointer, &entry->size, 8);
        pointer += 8;
        int len = entry->size * entry->ItemSize();
        if (len) {
            memcpy(pointer, entry->pointer, len);
        }
        pointer += len;
    }

    if ((pointer - buf) != size) {
        std::cout << "Invalid encoded size: " << (pointer - buf) << std::endl;
        return nullptr;
    }

    return buf;
}

std::size_t DamEncoder::CalculateSize() {
    auto size = kPrefixLen;

    for (auto entry : *entries) {
        size += 16;  // The Type and Len
        auto len = entry->size * entry->ItemSize();
        size += align(len);
    }

    return size;
}


// DamDecoder ======

DamDecoder::DamDecoder(std::uint8_t *buffer, std::size_t size, bool local_version) {
    this->local_version = local_version;
    this->buffer = buffer;
    this->buf_size = size;
    this->pos = buffer + kPrefixLen;
    if (size >= kPrefixLen) {
        memcpy(&len, buffer + 8, 8);
        memcpy(&data_set_id, buffer + 16, 8);
    } else {
        len = 0;
        data_set_id = 0;
    }
}

bool DamDecoder::IsValid() {
    auto sig = local_version ? kSignatureLocal : kSignature;
    return buf_size >= kPrefixLen && memcmp(buffer, sig, strlen(sig)) == 0;
}

std::vector<int64_t> DamDecoder::DecodeIntArray() {
    auto type = *reinterpret_cast<int64_t *>(pos);
    if (type != kDataTypeIntArray) {
        std::cout << "Data type " << type << " doesn't match Int Array" << std::endl;
        return std::vector<int64_t>();
    }
    pos += 8;

    auto len = *reinterpret_cast<int64_t *>(pos);
    pos += 8;
    auto ptr = reinterpret_cast<int64_t *>(pos);
    pos += align(8*len);
    return std::vector<int64_t>(ptr, ptr + len);
}

std::vector<double> DamDecoder::DecodeFloatArray() {
    auto type = *reinterpret_cast<int64_t *>(pos);
    if (type != kDataTypeFloatArray) {
        std::cout << "Data type " << type << " doesn't match Float Array" << std::endl;
        return std::vector<double>();
    }
    pos += 8;

    auto len = *reinterpret_cast<int64_t *>(pos);
    pos += 8;

    auto ptr = reinterpret_cast<double *>(pos);
    pos += align(8*len);
    return std::vector<double>(ptr, ptr + len);
}

void *DamDecoder::DecodeBytes(size_t *size) {
    auto type = *reinterpret_cast<int64_t *>(pos);
    if (type != kDataTypeBytes) {
        std::cout << "Data type " << type << " doesn't match bytes" << std::endl;
        return nullptr;
    }
    pos += 8;

    *size = *reinterpret_cast<int64_t *>(pos);
    pos += 8;

    if (*size == 0) {
        return nullptr;
    }

    auto ptr = reinterpret_cast<void *>(pos);
    pos += align(*size);
    return ptr;
}
