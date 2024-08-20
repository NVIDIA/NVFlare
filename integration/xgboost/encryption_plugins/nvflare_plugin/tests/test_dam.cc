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
#include "gtest/gtest.h"
#include "dam.h"

TEST(DamTest, TestEncodeDecode) {
    double float_array[] = {1.1, 1.2, 1.3, 1.4};
    int64_t int_array[] = {123, 456, 789};
    char buf1[] = "short";
    char buf2[] = "very long";


    DamEncoder encoder(123);
    auto b1 = Buffer(buf1, strlen(buf1));
    auto b2 = Buffer(buf2, strlen(buf2));
    encoder.AddBuffer(b1);
    encoder.AddBuffer(b2);

    std::vector<Buffer> b{b1, b2};
    encoder.AddBufferArray(b);

    auto f = std::vector<double>(float_array, float_array + 4);
    encoder.AddFloatArray(f);

    auto i = std::vector<int64_t>(int_array, int_array + 3);
    encoder.AddIntArray(i);

    size_t size;
    auto buf = encoder.Finish(size);
    std::cout << "Encoded size is " << size << std::endl;

    // Decoding test
    DamDecoder decoder(buf, size);
    EXPECT_EQ(decoder.IsValid(), true);
    EXPECT_EQ(decoder.GetDataSetId(), 123);

    auto new_buf1 = decoder.DecodeBuffer();
    EXPECT_EQ(0, memcmp(new_buf1.buffer, buf1, new_buf1.buf_size));

    auto new_buf2 = decoder.DecodeBuffer();
    EXPECT_EQ(0, memcmp(new_buf2.buffer, buf2, new_buf2.buf_size));

    auto buf_vec = decoder.DecodeBufferArray();
    EXPECT_EQ(2, buf_vec.size());
    EXPECT_EQ(0, memcmp(buf_vec[0].buffer, buf1, buf_vec[0].buf_size));
    EXPECT_EQ(0, memcmp(buf_vec[1].buffer, buf2, buf_vec[1].buf_size));

    auto float_vec = decoder.DecodeFloatArray();
    EXPECT_EQ(0, memcmp(float_vec.data(), float_array, float_vec.size()*8));

    auto int_vec = decoder.DecodeIntArray();
    EXPECT_EQ(0, memcmp(int_vec.data(), int_array, int_vec.size()*8));
}
