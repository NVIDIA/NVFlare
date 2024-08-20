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

template<class params>
struct implementation<test_barrett_rem_wide_1, params> {
  static const uint32_t TPI=params::TPI;
  static const uint32_t BITS=params::BITS;

  typedef cgbn_context_t<TPI, params>    context_t;
  typedef cgbn_env_t<context_t, BITS>    env_t;
  typedef typename env_t::cgbn_t         bn_t;
  typedef typename env_t::cgbn_wide_t    bn_wide_t;

  public:
  __device__ __host__ static void run(typename types<params>::input_t *inputs, typename types<params>::output_t *outputs, int32_t instance) {
    context_t context(cgbn_print_monitor);
    env_t     env(context);
    bn_t      h1, h2, d, approx, r1;
    bn_wide_t xh;
    int32_t   compare;
    uint32_t  lz;

    cgbn_load(env, h1, &(inputs[instance].h1));
    cgbn_load(env, h2, &(inputs[instance].h2));
    cgbn_load(env, xh._low, &(inputs[instance].x1));

    compare=cgbn_compare(env, h1, h2);
    if(compare==0)
      return;

    if(compare>0) {
      cgbn_set(env, xh._high, h2);
      cgbn_set(env, d, h1);
    }
    else {
      cgbn_set(env, xh._high, h1);
      cgbn_set(env, d, h2);
    }

    if(!cgbn_equals_ui32(env, d, 0)) {
      lz=cgbn_barrett_approximation(env, approx, d);
      cgbn_barrett_rem_wide(env, r1, xh, d, approx, lz);
    }
    else
      cgbn_set_ui32(env, r1, 0);

    cgbn_store(env, &(outputs[instance].r1), r1);
  }
};


