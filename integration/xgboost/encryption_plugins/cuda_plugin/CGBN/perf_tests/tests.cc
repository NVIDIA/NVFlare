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

typedef enum {
  test_all,
  test_unknown,

  gt_add,
  gt_sub,
  gt_mul,
  gt_div_qr,
  gt_sqrt,
  gt_powm_odd,
  gt_mont_reduce,
  gt_gcd,
  gt_modinv,

  xt_add,
  xt_sub,
  xt_accumulate,
  xt_mul,
  xt_div_qr,
  xt_sqrt,
  xt_powm_odd,
  xt_mont_reduce,
  xt_gcd,
  xt_modinv,

} test_t;

#define GT_FIRST gt_add
#define GT_LAST  gt_modinv

#define XT_FIRST xt_add
#define XT_LAST  xt_modinv

test_t gt_parse(const char *name) {
  if(strcmp(name, "add")==0)
    return gt_add;
  else if(strcmp(name, "sub")==0)
    return gt_sub;
  else if(strcmp(name, "mul")==0)
    return gt_mul;
  else if(strcmp(name, "div_qr")==0)
    return gt_div_qr;
  else if(strcmp(name, "sqrt")==0)
    return gt_sqrt;
  else if(strcmp(name, "powm_odd")==0)
    return gt_powm_odd;
  else if(strcmp(name, "mont_reduce")==0)
    return gt_mont_reduce;
  else if(strcmp(name, "gcd")==0)
    return gt_gcd;
  else if(strcmp(name, "modinv")==0)
    return gt_modinv;
  return test_unknown;
}

test_t xt_parse(const char *name) {
  if(strcmp(name, "add")==0)
    return xt_add;
  else if(strcmp(name, "sub")==0)
    return xt_sub;
  else if(strcmp(name, "accumulate")==0)
    return xt_accumulate;
  else if(strcmp(name, "mul")==0)
    return xt_mul;
  else if(strcmp(name, "div_qr")==0)
    return xt_div_qr;
  else if(strcmp(name, "sqrt")==0)
    return xt_sqrt;
  else if(strcmp(name, "powm_odd")==0)
    return xt_powm_odd;
  else if(strcmp(name, "mont_reduce")==0)
    return xt_mont_reduce;
  else if(strcmp(name, "gcd")==0)
    return xt_gcd;
  else if(strcmp(name, "modinv")==0)
    return xt_modinv;
  return test_unknown;
}

const char *test_name(test_t test) {
  switch(test) {
    case gt_add: case xt_add:
      return "add";
    case gt_sub: case xt_sub:
      return "sub";
    case xt_accumulate:
      return "accumulate";
    case gt_mul: case xt_mul:
      return "mul";
    case gt_div_qr: case xt_div_qr:
      return "div_qr";
    case gt_sqrt: case xt_sqrt:
      return "sqrt";
    case gt_powm_odd: case xt_powm_odd:
      return "powm_odd";
    case gt_mont_reduce: case xt_mont_reduce:
      return "mont_reduce";
    case gt_gcd: case xt_gcd:
      return "gcd";
    case gt_modinv: case xt_modinv:
      return "modinv";
  }
  return "unknown";
}


