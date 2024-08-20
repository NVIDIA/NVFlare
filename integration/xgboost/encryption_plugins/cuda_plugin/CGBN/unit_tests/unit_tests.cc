/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

// uncomment the next line to enable a full test at many sizes from 32 bits through 32K bits.  The full test is MUCH slower to compile and run.
// #define FULL_TEST

template<class T>
class CGBN1 : public testing::Test {
  public:
  static const uint32_t TPB=T::TPB;
  static const uint32_t MAX_ROTATION=T::MAX_ROTATION;
  static const uint32_t SHM_LIMIT=T::SHM_LIMIT;
  static const bool     CONSTANT_TIME=T::CONSTANT_TIME;
  
  static const uint32_t BITS=T::BITS;
  static const uint32_t TPI=T::TPI;

  static const uint32_t size=T::BITS;
};

template<class T>
class CGBN2 : public testing::Test {
  public:
  static const uint32_t TPB=T::TPB;
  static const uint32_t MAX_ROTATION=T::MAX_ROTATION;
  static const uint32_t SHM_LIMIT=T::SHM_LIMIT;
  static const bool     CONSTANT_TIME=T::CONSTANT_TIME;

  static const uint32_t BITS=T::BITS;
  static const uint32_t TPI=T::TPI;

  static const uint32_t size=T::BITS;
};

template<class T>
class CGBN3 : public testing::Test {
  public:
  static const uint32_t TPB=T::TPB;
  static const uint32_t MAX_ROTATION=T::MAX_ROTATION;
  static const uint32_t SHM_LIMIT=T::SHM_LIMIT;
  static const bool     CONSTANT_TIME=T::CONSTANT_TIME;

  static const uint32_t BITS=T::BITS;
  static const uint32_t TPI=T::TPI;

  static const uint32_t size=T::BITS;
};

template<class T>
class CGBN4 : public testing::Test {
  public:
  static const uint32_t TPB=T::TPB;
  static const uint32_t MAX_ROTATION=T::MAX_ROTATION;
  static const uint32_t SHM_LIMIT=T::SHM_LIMIT;
  static const bool     CONSTANT_TIME=T::CONSTANT_TIME;

  static const uint32_t BITS=T::BITS;
  static const uint32_t TPI=T::TPI;

  static const uint32_t size=T::BITS;
};

template<class T>
class CGBN5 : public testing::Test {
  public:
  static const uint32_t TPB=T::TPB;
  static const uint32_t MAX_ROTATION=T::MAX_ROTATION;
  static const uint32_t SHM_LIMIT=T::SHM_LIMIT;
  static const bool     CONSTANT_TIME=T::CONSTANT_TIME;

  static const uint32_t BITS=T::BITS;
  static const uint32_t TPI=T::TPI;

  static const uint32_t size=T::BITS;
};

TYPED_TEST_SUITE_P(CGBN1);
TYPED_TEST_SUITE_P(CGBN2);
TYPED_TEST_SUITE_P(CGBN3);
TYPED_TEST_SUITE_P(CGBN4);
TYPED_TEST_SUITE_P(CGBN5);

TYPED_TEST_P(CGBN1, set_1) {
  bool result=run_test<test_set_1, TestFixture>(LONG_TEST);

  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, swap_1) {
  bool result=run_test<test_swap_1, TestFixture>(LONG_TEST);

  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, add_1) {
  bool result=run_test<test_add_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, sub_1) {
  bool result=run_test<test_sub_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, negate_1) {
  bool result=run_test<test_negate_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, mul_1) {
  bool result=run_test<test_mul_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, mul_high_1) {
  bool result=run_test<test_mul_high_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, sqr_1) {
  bool result=run_test<test_sqr_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, sqr_high_1) {
  bool result=run_test<test_sqr_high_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, div_1) {
  bool result=run_test<test_div_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, rem_1) {
  bool result=run_test<test_rem_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, div_rem_1) {
  bool result=run_test<test_div_rem_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, sqrt_1) {
  bool result=run_test<test_sqrt_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, sqrt_rem_1) {
  bool result=run_test<test_sqrt_rem_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, equals_1) {
  bool result=run_test<test_equals_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, equals_2) {
  bool result=run_test<test_equals_2, TestFixture>(TINY_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, equals_3) {
  bool result=run_test<test_equals_3, TestFixture>(TINY_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, compare_1) {
  bool result=run_test<test_compare_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, compare_2) {
  bool result=run_test<test_compare_2, TestFixture>(TINY_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, compare_3) {
  bool result=run_test<test_compare_3, TestFixture>(TINY_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, compare_4) {
  bool result=run_test<test_compare_4, TestFixture>(TINY_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, extract_bits_1) {
  bool result=run_test<test_extract_bits_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, insert_bits_1) {
  bool result=run_test<test_insert_bits_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, get_ui32_set_ui32_1) {
  bool result=run_test<test_get_ui32_set_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, add_ui32_1) {
  bool result=run_test<test_add_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, sub_ui32_1) {
  bool result=run_test<test_sub_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, mul_ui32_1) {
  bool result=run_test<test_mul_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, div_ui32_1) {
  bool result=run_test<test_div_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, rem_ui32_1) {
  bool result=run_test<test_rem_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, equals_ui32_1) {
  bool result=run_test<test_equals_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, equals_ui32_2) {
  bool result=run_test<test_equals_ui32_2, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, equals_ui32_3) {
  bool result=run_test<test_equals_ui32_3, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, equals_ui32_4) {
  bool result=run_test<test_equals_ui32_4, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, compare_ui32_1) {
  bool result=run_test<test_compare_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, compare_ui32_2) {
  bool result=run_test<test_compare_ui32_2, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, extract_bits_ui32_1) {
  bool result=run_test<test_extract_bits_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, insert_bits_ui32_1) {
  bool result=run_test<test_insert_bits_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, binary_inverse_ui32_1) {
  bool result=run_test<test_binary_inverse_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, gcd_ui32_1) {
  bool result=run_test<test_gcd_ui32_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, mul_wide_1) {
  bool result=run_test<test_mul_wide_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, sqr_wide_1) {
  bool result=run_test<test_sqr_wide_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, div_wide_1) {
  bool result=run_test<test_div_wide_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, rem_wide_1) {
  bool result=run_test<test_rem_wide_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, div_rem_wide_1) {
  bool result=run_test<test_div_rem_wide_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, sqrt_wide_1) {
  bool result=run_test<test_sqrt_wide_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, sqrt_rem_wide_1) {
  bool result=run_test<test_sqrt_rem_wide_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_and_1) {
  bool result=run_test<test_bitwise_and_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_ior_1) {
  bool result=run_test<test_bitwise_ior_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_xor_1) {
  bool result=run_test<test_bitwise_xor_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}


TYPED_TEST_P(CGBN4, bitwise_complement_1) {
  bool result=run_test<test_bitwise_complement_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_select_1) {
  bool result=run_test<test_bitwise_select_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_mask_copy_1) {
  bool result=run_test<test_bitwise_mask_copy_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_mask_and_1) {
  bool result=run_test<test_bitwise_mask_and_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_mask_ior_1) {
  bool result=run_test<test_bitwise_mask_ior_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_mask_xor_1) {
  bool result=run_test<test_bitwise_mask_xor_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_mask_select_1) {
  bool result=run_test<test_bitwise_mask_select_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, shift_left_1) {
  bool result=run_test<test_shift_left_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, shift_right_1) {
  bool result=run_test<test_shift_right_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, rotate_left_1) {
  bool result=run_test<test_rotate_left_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, rotate_right_1) {
  bool result=run_test<test_rotate_right_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, pop_count_1) {
  bool result=run_test<test_pop_count_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, clz_1) {
  bool result=run_test<test_clz_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, ctz_1) {
  bool result=run_test<test_ctz_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, accumulator_1) {
  bool result=run_test<test_accumulator_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, accumulator_2) {
  bool result=run_test<test_accumulator_2, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, binary_inverse_1) {
  bool result=run_test<test_binary_inverse_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, gcd_1) {
  bool result=run_test<test_gcd_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, modular_inverse_1) {
  bool result=run_test<test_modular_inverse_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, modular_power_1) {
  bool result=run_test<test_modular_power_1, TestFixture>(MEDIUM_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, bn2mont_1) {
  bool result=run_test<test_bn2mont_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, mont2bn_1) {
  bool result=run_test<test_mont2bn_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, mont_mul_1) {
  bool result=run_test<test_mont_mul_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, mont_sqr_1) {
  bool result=run_test<test_mont_sqr_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, mont_reduce_wide_1) {
  bool result=run_test<test_mont_reduce_wide_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_div_1) {
  bool result=run_test<test_barrett_div_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_rem_1) {
  bool result=run_test<test_barrett_rem_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_div_rem_1) {
  bool result=run_test<test_barrett_div_rem_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_div_wide_1) {
  bool result=run_test<test_barrett_div_wide_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_rem_wide_1) {
  bool result=run_test<test_barrett_rem_wide_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_div_rem_wide_1) {
  bool result=run_test<test_barrett_div_rem_wide_1, TestFixture>(LONG_TEST);
  
  EXPECT_TRUE(result);
}

REGISTER_TYPED_TEST_SUITE_P(CGBN1,
 set_1, swap_1, add_1, sub_1, negate_1,
 mul_1, mul_high_1, sqr_1, sqr_high_1, div_1, rem_1, div_rem_1, sqrt_1,
 sqrt_rem_1, equals_1, equals_2, equals_3, compare_1, compare_2, compare_3, compare_4,
 extract_bits_1, insert_bits_1
);
REGISTER_TYPED_TEST_SUITE_P(CGBN2,
 get_ui32_set_ui32_1, add_ui32_1, sub_ui32_1, mul_ui32_1, div_ui32_1, rem_ui32_1,
 equals_ui32_1, equals_ui32_2, equals_ui32_3, equals_ui32_4, compare_ui32_1, compare_ui32_2,
 extract_bits_ui32_1, insert_bits_ui32_1, binary_inverse_ui32_1, gcd_ui32_1
);
REGISTER_TYPED_TEST_SUITE_P(CGBN3,
 mul_wide_1, sqr_wide_1, div_wide_1, rem_wide_1, div_rem_wide_1, sqrt_wide_1, sqrt_rem_wide_1
);
REGISTER_TYPED_TEST_SUITE_P(CGBN4,
 bitwise_and_1, bitwise_ior_1, bitwise_xor_1, bitwise_complement_1, bitwise_select_1, 
 bitwise_mask_copy_1, bitwise_mask_and_1, bitwise_mask_ior_1, bitwise_mask_xor_1, bitwise_mask_select_1,
 shift_left_1, shift_right_1, rotate_left_1, rotate_right_1, pop_count_1, clz_1, ctz_1
);
REGISTER_TYPED_TEST_SUITE_P(CGBN5,
 accumulator_1, accumulator_2, binary_inverse_1, gcd_1, modular_inverse_1, modular_power_1,
 bn2mont_1, mont2bn_1, mont_mul_1, mont_sqr_1, mont_reduce_wide_1, barrett_div_1, barrett_rem_1,
 barrett_div_rem_1, barrett_div_wide_1, barrett_rem_wide_1, barrett_div_rem_wide_1
);

INSTANTIATE_TYPED_TEST_SUITE_P(S32T4, CGBN1, size32t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S32T4, CGBN2, size32t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S32T4, CGBN3, size32t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S32T4, CGBN4, size32t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S32T4, CGBN5, size32t4);

#ifdef FULL_TEST
INSTANTIATE_TYPED_TEST_SUITE_P(S64T4, CGBN1, size64t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S64T4, CGBN2, size64t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S64T4, CGBN3, size64t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S64T4, CGBN4, size64t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S64T4, CGBN5, size64t4);

INSTANTIATE_TYPED_TEST_SUITE_P(S96T4, CGBN1, size96t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S96T4, CGBN2, size96t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S96T4, CGBN3, size96t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S96T4, CGBN4, size96t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S96T4, CGBN5, size96t4);
#endif

INSTANTIATE_TYPED_TEST_SUITE_P(S128T4, CGBN1, size128t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S128T4, CGBN2, size128t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S128T4, CGBN3, size128t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S128T4, CGBN4, size128t4);
INSTANTIATE_TYPED_TEST_SUITE_P(S128T4, CGBN5, size128t4);

INSTANTIATE_TYPED_TEST_SUITE_P(S192T8, CGBN1, size192t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S192T8, CGBN2, size192t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S192T8, CGBN3, size192t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S192T8, CGBN4, size192t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S192T8, CGBN5, size192t8);

INSTANTIATE_TYPED_TEST_SUITE_P(S256T8, CGBN1, size256t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S256T8, CGBN2, size256t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S256T8, CGBN3, size256t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S256T8, CGBN4, size256t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S256T8, CGBN5, size256t8);

INSTANTIATE_TYPED_TEST_SUITE_P(S288T8, CGBN1, size288t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S288T8, CGBN2, size288t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S288T8, CGBN3, size288t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S288T8, CGBN4, size288t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S288T8, CGBN5, size288t8);

INSTANTIATE_TYPED_TEST_SUITE_P(S512T8, CGBN1, size512t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S512T8, CGBN2, size512t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S512T8, CGBN3, size512t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S512T8, CGBN4, size512t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S512T8, CGBN5, size512t8);

INSTANTIATE_TYPED_TEST_SUITE_P(S1024T8, CGBN1, size1024t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T8, CGBN2, size1024t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T8, CGBN3, size1024t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T8, CGBN4, size1024t8);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T8, CGBN5, size1024t8);

#ifdef FULL_TEST
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T16, CGBN1, size1024t16);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T16, CGBN2, size1024t16);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T16, CGBN3, size1024t16);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T16, CGBN4, size1024t16);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T16, CGBN5, size1024t16);

INSTANTIATE_TYPED_TEST_SUITE_P(S1024T32, CGBN1, size1024t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T32, CGBN2, size1024t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T32, CGBN3, size1024t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T32, CGBN4, size1024t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S1024T32, CGBN5, size1024t32);
#endif

INSTANTIATE_TYPED_TEST_SUITE_P(S2048T32, CGBN1, size2048t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S2048T32, CGBN2, size2048t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S2048T32, CGBN3, size2048t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S2048T32, CGBN4, size2048t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S2048T32, CGBN5, size2048t32);

INSTANTIATE_TYPED_TEST_SUITE_P(S3072T32, CGBN1, size3072t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S3072T32, CGBN2, size3072t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S3072T32, CGBN3, size3072t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S3072T32, CGBN4, size3072t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S3072T32, CGBN5, size3072t32);

INSTANTIATE_TYPED_TEST_SUITE_P(S4096T32, CGBN1, size4096t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S4096T32, CGBN2, size4096t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S4096T32, CGBN3, size4096t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S4096T32, CGBN4, size4096t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S4096T32, CGBN5, size4096t32);

INSTANTIATE_TYPED_TEST_SUITE_P(S8192T32, CGBN1, size8192t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S8192T32, CGBN2, size8192t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S8192T32, CGBN3, size8192t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S8192T32, CGBN4, size8192t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S8192T32, CGBN5, size8192t32);

#ifdef FULL_TEST
INSTANTIATE_TYPED_TEST_SUITE_P(S16384T32, CGBN1, size16384t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S16384T32, CGBN2, size16384t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S16384T32, CGBN3, size16384t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S16384T32, CGBN4, size16384t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S16384T32, CGBN5, size16384t32);

INSTANTIATE_TYPED_TEST_SUITE_P(S32768T32, CGBN1, size32768t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S32768T32, CGBN2, size32768t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S32768T32, CGBN3, size32768t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S32768T32, CGBN4, size32768t32);
INSTANTIATE_TYPED_TEST_SUITE_P(S32768T32, CGBN5, size32768t32);
#endif
