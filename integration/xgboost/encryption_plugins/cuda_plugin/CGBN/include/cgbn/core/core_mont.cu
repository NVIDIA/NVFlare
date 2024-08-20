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

/* single limb version */
template<class env>
__device__ __forceinline__ void core_t<env>::mont_mul(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t n, const uint32_t np0) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t r0=0, r1=0, r2, t, c;
  
  #pragma unroll
  for(int32_t thread=0;thread<TPI;thread++) {
    // broadcast b[i]
    t=__shfl_sync(sync, b, thread, TPI);

    r0=madlo_cc(a, t, r0);
    r1=madhic_cc(a, t, r1);
    r2=addc(0, 0);

    // broadcast r[0]
    t=__shfl_sync(sync, r0, 0, TPI)*np0;
    r0=madlo_cc(n, t, r0);
    r1=madhic_cc(n, t, r1);
    r2=addc(r2, 0);
    
    // shift right by 32 bits (top thread gets zero)
    r0=__shfl_sync(sync, r0, threadIdx.x+1, TPI);
    r0=add_cc(r0, r1);
    r1=addc(r2, 0);
  }
 
  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, r1, 1, TPI);
  
  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  r0=add_cc(r0, t);
  c=addc(r1, 0);

  c=fast_propagate_add(c, r0);
  
  // compute -n
  t=n-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple
  t=~t & -c;

  r0=add_cc(r0, t);
  c=addc(0, 0);
  fast_propagate_add(c, r0);

  r=r0; 
}

template<class env>
__device__ __forceinline__ void core_t<env>::mont_reduce_wide(uint32_t &r, const uint32_t lo, const uint32_t hi, const uint32_t n, const uint32_t np0, const bool zero) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t r0=lo, r1=0, t, top;
  
  #pragma unroll
  for(int32_t thread=0;thread<TPI;thread++) {
    t=__shfl_sync(sync, r0, 0, TPI)*np0;
    r0=madlo_cc(n, t, r0);
    r1=madhic_cc(n, t, r1);
        
    // shift right by 32 bits (top thread gets zero)
    r0=__shfl_sync(sync, r0, threadIdx.x+1, TPI);
    if(!zero) {
      top=__shfl_sync(sync, hi, thread, TPI);
      r0=(group_thread==TPI-1) ? top : r0;
    }
    
    // add it in
    r0=add_cc(r0, r1);
    r1=addc(0, 0);
  }
 
  r1=fast_propagate_add(r1, r0);

  if(!zero && r1!=0) {
    // compute -n
    t=n-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple
    t=~t & -r1;

    r0=add_cc(r0, t);
    r1=addc(0, 0);
    fast_propagate_add(r1, r0);
  }
  r=r0; 
}

} /* namespace cgbn */

