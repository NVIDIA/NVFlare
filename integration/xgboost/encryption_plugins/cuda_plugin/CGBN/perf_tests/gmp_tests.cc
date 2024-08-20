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

extern "C" void __gmpn_binvert(mp_limb_t *i, const mp_limb_t *src, mp_size_t size, mp_limb_t *scratch);
extern "C" void __gmpn_redc_2(mp_limb_t *r, const mp_limb_t *src, const mp_limb_t *mod, mp_size_t size, const mp_limb_t *i);

uint64_t g_test_add(uint32_t bits, g_data_t *data, uint32_t count) {
  uint64_t LOOPS=40000;

  #pragma omp parallel for
  for(int index=0;index<count;index++) {
    mpz_t priv_t;

    mpz_init(priv_t);
    for(int loop=0;loop<LOOPS;loop++) {
      mpz_add(priv_t, priv_t, data[index].x0);
    }
    mpz_set(data[index].r, priv_t);
    mpz_clear(priv_t);
  }
  return LOOPS*count;
}

uint64_t g_test_sub(uint32_t bits, g_data_t *data, uint32_t count) {
  uint64_t LOOPS=40000;

  #pragma omp parallel for
  for(int index=0;index<count;index++) {
    mpz_t priv_t;

    mpz_init(priv_t);
    for(int loop=0;loop<LOOPS;loop++) {
      mpz_sub(priv_t, priv_t, data[index].x0);
    }
    mpz_set(data[index].r, priv_t);
    mpz_clear(priv_t);
  }
  return LOOPS*count;
}

uint64_t g_test_mul(uint32_t bits, g_data_t *data, uint32_t count) {
  uint64_t  LOOPS=500*8192/bits;
  mp_size_t len;

  #pragma omp parallel for
  for(int index=0;index<count;index++) {
    mpz_t priv_t;

    len=mpz_size(data[index].x0);
    mpz_init(priv_t);
    mpz_set(priv_t, data[index].x0);
    for(int loop=0;loop<LOOPS;loop++) {
      mpz_mul(priv_t, priv_t, data[index].x1);
      mpz_limbs_finish(priv_t, len);      // fast way to truncate product
    }
    mpz_set(data[index].r, priv_t);
  }
  return LOOPS*count;
}

uint64_t g_test_div_qr(uint32_t bits, g_data_t *data, uint32_t count) {
  uint64_t  LOOPS=500*8192/bits;
  mp_size_t len;

  #pragma omp parallel for
  for(int index=0;index<count;index++) {
    mpz_t priv_q, priv_r, priv_t;

    mpz_init(priv_q);
    mpz_init(priv_r);
    mpz_init(priv_t);
    mpz_set(priv_t, data[index].x0);
    for(int loop=0;loop<LOOPS;loop++) {
      mpz_fdiv_qr(priv_q, priv_r, priv_t, data[index].x1);
      mpz_mul_2exp(priv_t, priv_r, bits);
      mpz_add(priv_t, priv_t, priv_q);
    }
    mpz_set(data[index].r, priv_r);
  }
  return LOOPS*count;
}

uint64_t g_test_sqrt(uint32_t bits, g_data_t *data, uint32_t count) {
  uint64_t LOOPS=500*8192/bits;

  #pragma omp parallel for
  for(int index=0;index<count;index++) {
    mpz_t priv_t;

    mpz_init(priv_t);
    for(int loop=0;loop<LOOPS;loop++) {
      mpz_sqrt(priv_t, data[index].w0);
    }
    mpz_set(data[index].r, priv_t);
    mpz_clear(priv_t);
  }
  return LOOPS*count;
}

uint64_t g_test_powm_odd(uint32_t bits, g_data_t *data, uint32_t count) {
  uint64_t LOOPS=8192/bits;

  #pragma omp parallel for
  for(int index=0;index<count;index++) {
    mpz_t priv_t;

    mpz_init(priv_t);
    for(int loop=0;loop<LOOPS;loop++) {
      mpz_powm(priv_t, data[index].x0, data[index].s0, data[index].o0);
    }
    mpz_set(data[index].r, priv_t);
    mpz_clear(priv_t);
  }
  return LOOPS*count;
}

uint64_t g_test_mont_reduce(uint32_t bits, g_data_t *data, uint32_t count) {
  uint64_t LOOPS=500*8192/bits;

  #pragma omp parallel for
  for(int index=0;index<count;index++) {
    mpz_t           priv_t;
    mp_limb_t       *r_limbs;
    const mp_limb_t *a_limbs, *n_limbs;
    mp_limb_t       i[2];
    mp_limb_t       scratch[128];

    mpz_init2(priv_t, bits);
    r_limbs=mpz_limbs_write(priv_t, bits/64);
    a_limbs=mpz_limbs_read(data[index].w0);
    n_limbs=mpz_limbs_read(data[index].o0);

    __gmpn_binvert(i, mpz_limbs_read(data[index].o0), 2, scratch);
    i[0]=-i[0];
    i[1]=~i[1];

    for(int word=0;word<bits/64;word++)
      r_limbs[word]=0;

    for(int loop=0;loop<LOOPS;loop++) {
      __gmpn_redc_2(r_limbs, a_limbs, n_limbs, bits/64, i);
    }
    int32_t count=bits/64;
    while(count>0 && r_limbs[count-1]==0)
      count--;
    mpz_limbs_finish(priv_t, count);
    mpz_set(data[index].r, priv_t);
  }
  return LOOPS*count;
}

uint64_t g_test_gcd(uint32_t bits, g_data_t *data, uint32_t count) {
  uint64_t LOOPS=50*8192/bits;

  #pragma omp parallel for
  for(int index=0;index<count;index++) {
    mpz_t priv_t;

    mpz_init(priv_t);
    for(int loop=0;loop<LOOPS;loop++) {
      mpz_gcd(priv_t, data[index].x0, data[index].x1);
    }
    mpz_set(data[index].r, priv_t);
  }
  return LOOPS*count;
}

uint64_t g_test_modinv(uint32_t bits, g_data_t *data, uint32_t count) {
  uint64_t LOOPS=50*8192/bits;

  #pragma omp parallel for
  for(int index=0;index<count;index++) {
    mpz_t priv_t;

    mpz_init(priv_t);
    for(int loop=0;loop<LOOPS;loop++) {
      mpz_invert(priv_t, data[index].x0, data[index].x1);
    }
    mpz_set(data[index].r, priv_t);
  }
  return LOOPS*count;
}

