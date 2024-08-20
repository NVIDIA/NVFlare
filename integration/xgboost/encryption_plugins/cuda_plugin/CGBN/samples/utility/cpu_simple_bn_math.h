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

void add_words(uint32_t *r, uint32_t *x, uint32_t *y, uint32_t count) {
  int     index;
  int64_t sum=0;

  for(index=0;index<count;index++) {
    sum=sum+x[index]+y[index];
    r[index]=sum;
    sum=sum>>32;
  }
}

void sub_words(uint32_t *r, uint32_t *x, uint32_t *y, uint32_t count) {
  int     index;
  int64_t sum=0;

  for(index=0;index<count;index++) {
    sum=sum+x[index]-y[index];
    r[index]=sum;
    sum=sum>>32;
  }
}

