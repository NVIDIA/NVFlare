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

class size32t4 {
  public:
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;   // SHM_LIMIT or SHARED_LIMIT
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=32;
  static const uint32_t TPI=4;
};

class size64t4 {
  public:
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;   // SHM_LIMIT or SHARED_LIMIT
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=64;
  static const uint32_t TPI=4;
};

class size96t4 {
  public:
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;   // SHM_LIMIT or SHARED_LIMIT
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=96;
  static const uint32_t TPI=4;
};

class size128t4 {
  public:
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;   // SHM_LIMIT or SHARED_LIMIT
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=128;
  static const uint32_t TPI=4;
};

class size192t8 {
  public:
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=192;
  static const uint32_t TPI=8;
};

class size256t8 {
  public:
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;   // SHM_LIMIT or SHARED_LIMIT
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=256;
  static const uint32_t TPI=8;
};

class size288t8 {
  public:
  // required parameters for a cgbn_parameters class
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=288;
  static const uint32_t TPI=8;
};

class size512t8 {
  public:
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=512;
  static const uint32_t TPI=8;
};

class size1024t8 {
  public:
  // required parameters for a cgbn_parameters class
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=1024;
  static const uint32_t TPI=8;
};

class size1024t16 {
  public:
  // required parameters for a cgbn_parameters class
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=1024;
  static const uint32_t TPI=16;
};

class size1024t32 {
  public:
  // required parameters for a cgbn_parameters class
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=1024;
  static const uint32_t TPI=32;
};

class size2048t32 {
  public:
  // required parameters for a cgbn_parameters class
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=2048;
  static const uint32_t TPI=32;
};

class size3072t32 {
  public:
  // required parameters for a cgbn_parameters class
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=3072;
  static const uint32_t TPI=32;
};

class size4096t32 {
  public:
  // required parameters for a cgbn_parameters class
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=4096;
  static const uint32_t TPI=32;
};

class size8192t32 {
  public:
  // required parameters for a cgbn_parameters class
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=8192;
  static const uint32_t TPI=32;
};

class size16384t32 {
  public:
  // required parameters for a cgbn_parameters class
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=16384;
  static const uint32_t TPI=32;
};

class size32768t32 {
  public:
  // required parameters for a cgbn_parameters class
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;

  static const uint32_t BITS=32768;
  static const uint32_t TPI=32;
};



