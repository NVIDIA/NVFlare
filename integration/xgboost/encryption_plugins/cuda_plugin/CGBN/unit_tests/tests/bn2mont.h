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
struct implementation<test_bn2mont_1, params> {
  static const uint32_t TPI=params::TPI;
  static const uint32_t BITS=params::BITS;

  typedef cgbn_context_t<TPI, params>    context_t;
  typedef cgbn_env_t<context_t, BITS>    env_t;
  typedef typename env_t::cgbn_t         bn_t;

  public:
  __device__ __host__ static void run(typename types<params>::input_t *inputs, typename types<params>::output_t *outputs, int32_t instance) {
    context_t context(cgbn_print_monitor);
    env_t     env(context);
    bn_t      h1, h2, x1, r1, r2, temp;
    int32_t   compare;

    cgbn_load(env, h1, &(inputs[instance].h1));
    cgbn_load(env, h2, &(inputs[instance].h2));
    cgbn_load(env, x1, &(inputs[instance].x1));

    compare=cgbn_compare(env, h1, h2);
    if(compare>0) {
      cgbn_set(env, temp, h1);
      cgbn_set(env, h1, h2);
      cgbn_set(env, h2, temp);
    }
    else if(compare==0)
      cgbn_set_ui32(env, h1, 0);

    cgbn_bitwise_mask_ior(env, h2, h2, 1);

    cgbn_set_ui32(env, r2, cgbn_bn2mont(env, r1, h1, h2));

    cgbn_store(env, &(outputs[instance].r1), r1);
    cgbn_store(env, &(outputs[instance].r2), r2);
  }
};

