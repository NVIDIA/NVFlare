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
void  DamEncoder::AddBytes(const void *buffer, const size_t buf_size) {
    std::cout << "AddBytes called, size:  " << buf_size << std::endl;
    if (encoded) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    // print_buffer(buffer, buf_size);
    entries->push_back(new Entry(kDataTypeBytes, reinterpret_cast<const uint8_t *>(buffer), buf_size));
}

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

void  DamEncoder::AddBytesArray(const std::vector<Buffer> &value) {
    std::cout << "AddBytesArray called, size:  " << value.size() << std::endl;
    if (encoded) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    size_t size = 0;
    for (auto &buf: value) {
        size += buf.buf_size;
    }
    size += 8*value.size();
    entries->push_back(new Entry(kDataTypeBytesArray, reinterpret_cast<const uint8_t *>(&value), size));
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
        int len;
        if (entry->data_type == kDataTypeBytesArray) {
            const std::vector<Buffer> *buffers = reinterpret_cast<const std::vector<Buffer> *>(entry->pointer);
            memcpy(pointer, &entry->data_type, 8);
            pointer += 8;
            int64_t array_size = buffers->size();
            memcpy(pointer, &array_size, 8);
            pointer += 8;
            int64_t *sizes = reinterpret_cast<int64_t *>(pointer);
            for (auto &buf : *buffers) {
                *sizes = buf.buf_size;
                sizes++;
            }
            len = 8*buffers->size();
            pointer += len;
            for (auto &buf : *buffers) {
                if (buf.buf_size > 0) {
                    memcpy(pointer, buf.buffer, buf.buf_size);
                }
                pointer += buf.buf_size;
                len += buf.buf_size;
            }

        } else {
            memcpy(pointer, &entry->data_type, 8);
            pointer += 8;
            memcpy(pointer, &entry->size, 8);
            pointer += 8;
            len = entry->size * entry->ItemSize();
            if (len) {
                memcpy(pointer, entry->pointer, len);
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

std::vector<int64_t> DamDecoder::DecodeIntArray() {
    auto type = *reinterpret_cast<int64_t *>(pos);
    if (type != kDataTypeIntArray) {
        std::cout << "Data type " << type << " doesn't match Int Array" << std::endl;
        return std::vector<int64_t>();
    }
    pos += 8;

    auto array_size = *reinterpret_cast<int64_t *>(pos);
    pos += 8;
    auto ptr = reinterpret_cast<int64_t *>(pos);
    pos += align(8*array_size);
    return std::vector<int64_t>(ptr, ptr + len);
}

std::vector<double> DamDecoder::DecodeFloatArray() {
    auto type = *reinterpret_cast<int64_t *>(pos);
    if (type != kDataTypeFloatArray) {
        std::cout << "Data type " << type << " doesn't match Float Array" << std::endl;
        return std::vector<double>();
    }
    pos += 8;

    auto array_size = *reinterpret_cast<int64_t *>(pos);
    pos += 8;

    auto ptr = reinterpret_cast<double *>(pos);
    pos += align(8*array_size);
    return std::vector<double>(ptr, ptr + len);
}

std::vector<Buffer> DamDecoder::DecodeBytesArray() {
    auto type = *reinterpret_cast<int64_t *>(pos);
    if (type != kDataTypeBytesArray) {
        std::cout << "Data type " << type << " doesn't match Bytes Array" << std::endl;
        return std::vector<Buffer>();
    }
    pos += 8;

    auto num = *reinterpret_cast<int64_t *>(pos);
    pos += 8;

    auto size_ptr = reinterpret_cast<int64_t *>(pos);
    auto buf_ptr = pos + 8 * num;
    size_t total_size = 8 * num;
    auto result = std::vector<Buffer>(num);
    for (int i = 0; i < num; i++) {
        auto size = size_ptr[i];
        if (buf_size > 0) {
            result[i].buf_size =size;
            result[i].buffer = buf_ptr;
            buf_ptr += size;
        }
        total_size += size;
    }

    pos += align(total_size);
    return result;
}
