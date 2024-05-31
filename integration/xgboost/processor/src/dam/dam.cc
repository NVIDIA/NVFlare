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


void print_hex(uint8_t *buffer, int size) {
    if (size )
        for (int i = 0; i < size; i++) {
            auto c = buffer[i];
            std::cout << std::hex << (int) c << " ";
        }
    std::cout << std::endl << std::dec;
}

void print_buffer(uint8_t *buffer, int size) {
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
    std::cout << "AddBuffer called, size:  " << buffer.buf_size << std::endl;
    if (encoded) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    // print_buffer(buffer, buf_size);
    entries->push_back(new Entry(kDataTypeBuffer, reinterpret_cast<const uint8_t *>(buffer.buffer), buffer.buf_size));
}

void DamEncoder::AddFloatArray(const std::vector<double> &value) {
    std::cout << "AddFloatArray called, size:  " << value.size() << std::endl;
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

void  DamEncoder::AddBufferArray(const std::vector<Buffer> &value) {
    std::cout << "AddBufferArray called, size:  " << value.size() << std::endl;
    if (encoded) {
        std::cout << "Buffer is already encoded" << std::endl;
        return;
    }
    size_t size = 0;
    for (auto &buf: value) {
        size += buf.buf_size;
    }
    size += 8*value.size();
    entries->push_back(new Entry(kDataTypeBufferArray, reinterpret_cast<const uint8_t *>(&value), size));
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
        if (entry->data_type == kDataTypeBufferArray) {
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
            auto buf_ptr = pointer + len;
            for (auto &buf : *buffers) {
                if (buf.buf_size > 0) {
                    memcpy(buf_ptr, buf.buffer, buf.buf_size);
                }
                buf_ptr += buf.buf_size;
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

Buffer DamDecoder::DecodeBuffer() {
    auto type = *reinterpret_cast<int64_t *>(pos);
    if (type != kDataTypeBuffer) {
        std::cout << "Data type " << type << " doesn't match bytes" << std::endl;
        return Buffer();
    }
    pos += 8;

    auto size = *reinterpret_cast<int64_t *>(pos);
    pos += 8;

    if (size == 0) {
        return Buffer();
    }

    auto ptr = reinterpret_cast<void *>(pos);
    pos += align(size);
    return Buffer(ptr, size);
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
    return std::vector<int64_t>(ptr, ptr + array_size);
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
    return std::vector<double>(ptr, ptr + array_size);
}

std::vector<Buffer> DamDecoder::DecodeBufferArray() {
    auto type = *reinterpret_cast<int64_t *>(pos);
    if (type != kDataTypeBufferArray) {
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
            result[i].buf_size = size;
            result[i].buffer = buf_ptr;
            buf_ptr += size;
        }
        total_size += size;
    }

    pos += align(total_size);
    return result;
}
