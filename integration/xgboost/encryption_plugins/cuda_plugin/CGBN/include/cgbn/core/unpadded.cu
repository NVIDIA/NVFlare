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

namespace cgbn {

template<class params>
class unpadded_t {
  public:
  static const uint32_t        TPB=params::TPB;
  static const uint32_t        TPI=params::TPI;
  static const uint32_t        LIMBS=params::LIMBS;
  static const uint32_t        BITS=TPI*LIMBS*32;
  static const uint32_t        PADDING=0;
  static const uint32_t        MAX_ROTATION=params::MAX_ROTATION;
  static const uint32_t        SHM_LIMIT=params::SHM_LIMIT;
  static const bool            CONSTANT_TIME=params::CONSTANT_TIME;
  static const cgbn_syncable_t SYNCABLE=params::SYNCABLE;  
};

} /* namespace cgbn */