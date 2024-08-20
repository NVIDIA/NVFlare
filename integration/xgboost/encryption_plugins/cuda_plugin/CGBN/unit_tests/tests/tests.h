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

#include "set.h"
#include "swap.h"
#include "extract_bits.h"
#include "insert_bits.h"
#include "add.h"
#include "sub.h"
#include "negate.h"
#include "mul.h"
#include "mul_high.h"
#include "sqr.h"
#include "sqr_high.h"
#include "div.h"
#include "rem.h"
#include "div_rem.h"
#include "sqrt.h"
#include "sqrt_rem.h"
#include "equals_1.h"
#include "equals_2.h"
#include "equals_3.h"
#include "compare_1.h"
#include "compare_2.h"
#include "compare_3.h"
#include "compare_4.h"

#include "get_ui32_set_ui32.h"
#include "add_ui32.h"
#include "sub_ui32.h"
#include "mul_ui32.h"
#include "div_ui32.h"
#include "rem_ui32.h"
#include "equals_ui32_1.h"
#include "equals_ui32_2.h"
#include "equals_ui32_3.h"
#include "equals_ui32_4.h"
#include "compare_ui32_1.h"
#include "compare_ui32_2.h"
#include "extract_bits_ui32.h"
#include "insert_bits_ui32.h"
#include "binary_inverse_ui32.h"
#include "gcd_ui32.h"

#include "mul_wide.h"
#include "sqr_wide.h"
#include "div_wide.h"
#include "rem_wide.h"
#include "div_rem_wide.h"
#include "sqrt_wide.h"
#include "sqrt_rem_wide.h"

#include "bitwise_and.h"
#include "bitwise_ior.h"
#include "bitwise_xor.h"
#include "bitwise_complement.h"
#include "bitwise_select.h"
#include "bitwise_mask_copy.h"
#include "bitwise_mask_and.h"
#include "bitwise_mask_ior.h"
#include "bitwise_mask_xor.h"
#include "bitwise_mask_select.h"
#include "shift_left.h"
#include "shift_right.h"
#include "rotate_left.h"
#include "rotate_right.h"
#include "pop_count.h"
#include "clz.h"
#include "ctz.h"

#include "accumulator_1.h"
#include "accumulator_2.h"
#include "binary_inverse.h"
#include "gcd.h"
#include "modular_inverse.h"
#include "bn2mont.h"
#include "mont2bn.h"
#include "mont_mul.h"
#include "mont_sqr.h"
#include "mont_reduce_wide.h"
#include "barrett_div.h"
#include "barrett_rem.h"
#include "barrett_div_rem.h"
#include "barrett_div_wide.h"
#include "barrett_rem_wide.h"
#include "barrett_div_rem_wide.h"
#include "modular_power.h"
