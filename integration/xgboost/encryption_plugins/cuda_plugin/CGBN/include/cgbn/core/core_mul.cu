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

template<class env>
__device__ __forceinline__ void core_t<env>::mul(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t rl, p0=add, p1=0, t;
  int32_t  threads=(PADDING!=0) ? PADDING : TPI;

  #pragma unroll
  for(int32_t index=0;index<threads;index++) {
    t=__shfl_sync(sync, b, index, TPI);

    p0=madlo_cc(a, t, p0);
    p1=addc(p1, 0);
    
    if(group_thread<threads-index) 
      rl=p0;

    rl=__shfl_sync(sync, rl, threadIdx.x+1, TPI);

    p0=madhi_cc(a, t, p1);
    p1=addc(0, 0);
    
    p0=add_cc(p0, rl);
    p1=addc(p1, 0);
  }
  if(PADDING==0)
    r=rl;
  else
    r=__shfl_sync(sync, rl, threadIdx.x-threads, TPI);
}

template<class env>
__device__ __forceinline__ void core_t<env>::mul_wide(uint32_t &lo, uint32_t &hi, const uint32_t a, const uint32_t b, const uint32_t add) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t p0=add, p1=0, rl, t;
  int32_t  threads=(PADDING!=0) ? PADDING : TPI;
    
  if(PADDING!=0)
    rl=0;
    
  #pragma unroll
  for(int32_t index=0;index<threads;index++) {
    t=__shfl_sync(sync, b, index, TPI);
    p0=madlo_cc(a, t, p0);
    p1=madhic(a, t, p1);
    
    t=__shfl_sync(sync, p0, 0, TPI);
    if(group_thread==index)
      rl=t;
    
    p0=__shfl_down_sync(sync, p0, 1, TPI);
    if(PADDING==0)
      p0=(group_thread==TPI-1) ? 0 : p0;
      
    p0=add_cc(p0, p1);
    p1=addc(0, 0);
  }

  lo=rl;
  hi=p0;
  fast_propagate_add(p1, hi);
}

} /* namespace cgbn */