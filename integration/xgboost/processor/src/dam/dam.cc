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

// DamEncoder ======
void DamEncoder::AddFloatArray(const std::vector<double> &value) {
    if (encoded) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    auto buf_size = value.size()*8;
    uint8_t *buffer = static_cast<uint8_t *>(malloc(buf_size));
    memcpy(buffer, value.data(), buf_size);
    // print_buffer(reinterpret_cast<uint8_t *>(value.data()), value.size() * 8);
    entries->push_back(new Entry(kDataTypeFloatArray, buffer, value.size()));
}

void  DamEncoder::AddIntArray(const std::vector<int64_t> &value) {
    std::cout << "AddIntArray called, size:  " << value.size() << std::endl;
    if (encoded) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    auto buf_size = value.size()*8;
    std::cout << "Allocating " << buf_size << " bytes" << std::endl;
    uint8_t *buffer = static_cast<uint8_t *>(malloc(buf_size));
    memcpy(buffer, value.data(), buf_size);
    // print_buffer(buffer, buf_size);
    entries->push_back(new Entry(kDataTypeIntArray, buffer, value.size()));
}

std::uint8_t * DamEncoder::Finish(size_t &size) {
    encoded = true;

    size = calculate_size();
    auto buf = static_cast<uint8_t *>(malloc(size));
    auto pointer = buf;
    memcpy(pointer, kSignature, strlen(kSignature));
    memcpy(pointer+8, &size, 8);
    memcpy(pointer+16, &data_set_id, 8);

    pointer += kPrefixLen;
    for (auto entry : *entries) {
        memcpy(pointer, &entry->data_type, 8);
        pointer += 8;
        memcpy(pointer, &entry->size, 8);
        pointer += 8;
        int len = 8*entry->size;
        memcpy(pointer, entry->pointer, len);
        free(entry->pointer);
        pointer += len;
        // print_buffer(entry->pointer, entry->size*8);
    }

    if ((pointer - buf) != size) {
        std::cout << "Invalid encoded size: " << (pointer - buf) << std::endl;
        return nullptr;
    }

    return buf;
}

std::size_t DamEncoder::calculate_size() {
    auto size = kPrefixLen;

    for (auto entry : *entries) {
        size += 16;  // The Type and Len
        size += entry->size * 8;  // All supported data types are 8 bytes
    }

    return size;
}


// DamDecoder ======

DamDecoder::DamDecoder(std::uint8_t *buffer, std::size_t size) {
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
    return buf_size >= kPrefixLen && memcmp(buffer, kSignature, strlen(kSignature)) == 0;
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
    pos += 8*len;
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
    pos += 8*len;
    return std::vector<double>(ptr, ptr + len);
}
