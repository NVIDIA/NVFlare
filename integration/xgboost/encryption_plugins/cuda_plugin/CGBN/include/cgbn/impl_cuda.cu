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

#include "arith/arith.h"
#include "core/unpadded.cu"
#include "core/core.cu"
#include "core/core_singleton.cu"

#if(__CUDACC_VER_MAJOR__<9 || (__CUDACC_VER_MAJOR__==9 && __CUDACC_VER_MINOR__<2))
  #if __CUDA_ARCH__>=700
    #error CGBN requires CUDA version 9.2 or above on Volta
  #endif
#endif

/****************************************************************************************************************
 * cgbn_context_t implementation for CUDA
 ****************************************************************************************************************/
template<uint32_t tpi, class params>
__device__ __forceinline__ cgbn_context_t<tpi, params>::cgbn_context_t() : _monitor(cgbn_no_checks), _report(NULL), _instance(0xFFFFFFFF) {
}

template<uint32_t tpi, class params>
__device__ __forceinline__ cgbn_context_t<tpi, params>::cgbn_context_t(cgbn_monitor_t monitor) : _monitor(monitor), _report(NULL), _instance(0xFFFFFFFF) {
  if(monitor!=cgbn_no_checks) {
    if(tpi!=32 && tpi!=16 && tpi!=8 && tpi!=4)
      report_error(cgbn_unsupported_threads_per_instance);
    if(params::TPB!=0 && params::TPB!=blockDim.x)
      report_error(cgbn_threads_per_block_mismatch);
    if(params::CONSTANT_TIME)
      report_error(cgbn_unsupported_operation);
  }
}

template<uint32_t tpi, class params>
__device__ __forceinline__ cgbn_context_t<tpi, params>::cgbn_context_t(cgbn_monitor_t monitor, cgbn_error_report_t *report) : _monitor(monitor), _report(report), _instance(0xFFFFFFFF) {
  if(monitor!=cgbn_no_checks) {
    if(tpi!=32 && tpi!=16 && tpi!=8 && tpi!=4)
      report_error(cgbn_unsupported_threads_per_instance);
    if(params::TPB!=0 && params::TPB!=blockDim.x)
      report_error(cgbn_threads_per_block_mismatch);
    if(params::CONSTANT_TIME)
      report_error(cgbn_unsupported_operation);
  }
}

template<uint32_t tpi, class params>
__device__ __forceinline__ cgbn_context_t<tpi, params>::cgbn_context_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, uint32_t instance) : _monitor(monitor), _report(report), _instance(instance) {
  if(monitor!=cgbn_no_checks) {
    if(tpi!=32 && tpi!=16 && tpi!=8 && tpi!=4)
      report_error(cgbn_unsupported_threads_per_instance);
    if(params::TPB!=0 && params::TPB!=blockDim.x)
      report_error(cgbn_threads_per_block_mismatch);
    if(params::CONSTANT_TIME)
      report_error(cgbn_unsupported_operation);
  }
}

template<uint32_t tpi, class params>
__device__ __forceinline__ bool cgbn_context_t<tpi, params>::check_errors() const {
  return _monitor!=cgbn_no_checks;
}

template<uint32_t tpi, class params>
__device__ __noinline__ void cgbn_context_t<tpi, params>::report_error(cgbn_error_t error) const {
  if((threadIdx.x & tpi-1)==0) {
    if(_report!=NULL) {
      if(atomicCAS((uint32_t *)&(_report->_error), (uint32_t)cgbn_no_error, (uint32_t)error)==cgbn_no_error) {
        _report->_instance=_instance;
        _report->_threadIdx=threadIdx;
        _report->_blockIdx=blockIdx;
      }
    }

    if(_monitor==cgbn_print_monitor) {
      switch(_report->_error) {
        case cgbn_unsupported_threads_per_instance:
          printf("cgbn error: unsupported threads per instance\n");
          break;
        case cgbn_unsupported_size:
          printf("cgbn error: unsupported size\n");
          break;
        case cgbn_unsupported_limbs_per_thread:
          printf("cgbn error: unsupported limbs per thread\n");
          break;
        case cgbn_unsupported_operation:
          printf("cgbn error: unsupported operation\n");
          break;
        case cgbn_threads_per_block_mismatch:
          printf("cgbn error: TPB does not match blockDim.x\n");
          break;
        case cgbn_threads_per_instance_mismatch:
          printf("cgbn errpr: TPI does not match env_t::TPI\n");
          break;
        case cgbn_division_by_zero_error:
          printf("cgbn error: division by zero on instance\n");
          break;
        case cgbn_division_overflow_error:
          printf("cgbn error: division overflow on instance\n");
          break;
        case cgbn_invalid_montgomery_modulus_error:
          printf("cgbn error: division invalid montgomery modulus\n");
          break;
        case cgbn_modulus_not_odd_error:
          printf("cgbn error: invalid modulus (it must be odd)\n");
          break;
        case cgbn_inverse_does_not_exist_error:
          printf("cgbn error: inverse does not exist\n");
          break;
        default:
          printf("cgbn error: unknown error reported by instance\n");
          break;
      }
    }
    else if(_monitor==cgbn_halt_monitor) {
      __trap();
    }
  }
}

/*
template<uint32_t threads_per_instance, uint32_t threads_per_block> template<uint32_t bits>
__device__ __forceinline__ cgbn_env_t<cgbn_context_t, bits> cgbn_context_t<threads_per_instance, threads_per_block>::env() {
  cgbn_env_t<cgbn_context_t, bits> env(this);

  return env;
}

template<uint32_t threads_per_instance, uint32_t threads_per_block> template<typename env_t>
  __device__ __forceinline__ cgbn_env_t<cgbn_context_t, env_t::_bits> cgbn_context_t<threads_per_instance, threads_per_block>::env() {
    return env<env_t::_bits>();
}
*/

/****************************************************************************************************************
 * cgbn_env_t implementation for CUDA
 ****************************************************************************************************************/

/* constructor */
template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ cgbn_env_t<context_t, bits, syncable>::cgbn_env_t(const context_t &context) : _context(context) {
  if(_context.check_errors()) {
    if(bits==0 || (bits & 0x1F)!=0) 
      _context.report_error(cgbn_unsupported_size);
  }
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::set(cgbn_t &r, const cgbn_t &a) const {
  cgbn::core_t<cgbn_env_t>::set(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::swap(cgbn_t &r, cgbn_t &a) const {
  cgbn::core_t<cgbn_env_t>::swap(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable> template<class source_cgbn_t>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::set(cgbn_t &r, const source_cgbn_t &source) const {
  uint32_t sync, group_thread=threadIdx.x & TPI-1;
  uint32_t source_thread=0, source_limb=0, value;

  // TPI and TPB must match.  TPB matches automatically
  if(_context.check_errors()) {
    if(TPI!=source_cgbn_t::parent_env_t::TPI) {
      _context.report_error(cgbn_threads_per_instance_mismatch);
      return;
    }
  }
  
  sync=cgbn::core_t<cgbn_env_t>::sync_mask();
  cgbn::mpzero<LIMBS>(r._limbs);
  #pragma nounroll
  for(int32_t index=0;index<BITS/32;index++) {
    #pragma unroll
    for(int32_t limb=0;limb<source_cgbn_t::parent_env_t::LIMBS;limb++)
      if(limb==source_limb)
        value=source._limbs[limb];
    value=__shfl_sync(sync, value, source_thread, TPI);
    #pragma unroll
    for(int32_t limb=0;limb<LIMBS;limb++)
      if(group_thread*LIMBS+limb==index)
        r._limbs[limb]=value;
    source_limb++;
    if(source_limb==source_cgbn_t::parent_env_t::LIMBS) {
      source_limb=0;
      if(++source_thread==TPI)
        break;
    }
  }
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::extract_bits(cgbn_t &r, const cgbn_t &a, const uint32_t start, const uint32_t len) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  
  uint32_t local_len=len;
  
  if(start>=BITS) {
    cgbn::mpzero<LIMBS>(r._limbs);
    return;
  }
  
  local_len=cgbn::umin(local_len, BITS-start);
  
  core::rotate_right(r._limbs, a._limbs, start);
  core::bitwise_mask_and(r._limbs, r._limbs, local_len);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::insert_bits(cgbn_t &r, const cgbn_t &a, const uint32_t start, const uint32_t len, const cgbn_t &value) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;

  uint32_t local_len=len;
  uint32_t mask[LIMBS], temp[LIMBS];
  
  if(start>=BITS) {
    cgbn::mpset<LIMBS>(r._limbs, a._limbs);
    return;
  }
  
  local_len=cgbn::umin(local_len, BITS-start);
  
  core::rotate_left(temp, value._limbs, start);
  core::bitwise_mask_copy(mask, start+local_len);
  core::bitwise_mask_xor(mask, mask, start);
  core::bitwise_select(r._limbs, a._limbs, temp, mask);
}

/* ui32 routines */
template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::get_ui32(const cgbn_t &a) const {
  return cgbn::core_t<cgbn_env_t>::get_ui32(a._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::set_ui32(cgbn_t &r, const uint32_t value) const {
  cgbn::core_t<cgbn_env_t>::set_ui32(r._limbs, value);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ int32_t cgbn_env_t<context_t, bits, syncable>::add_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t add) const {
  return cgbn::core_t<cgbn_env_t>::add_ui32(r._limbs, a._limbs, add);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ int32_t cgbn_env_t<context_t, bits, syncable>::sub_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t sub) const {
  return cgbn::core_t<cgbn_env_t>::sub_ui32(r._limbs, a._limbs, sub);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::mul_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t mul) const {
  return cgbn::core_t<cgbn_env_t>::mul_ui32(r._limbs, a._limbs, mul);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::div_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t div) const {
  if(div==0) {
    if(_context.check_errors()) 
      _context.report_error(cgbn_division_by_zero_error);
    return 0;
  }
  return cgbn::core_singleton_t<cgbn_env_t, LIMBS>::div_ui32(r._limbs, a._limbs, div);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::rem_ui32(const cgbn_t &a, const uint32_t div) const {
  if(div==0) {
    if(_context.check_errors()) 
      _context.report_error(cgbn_division_by_zero_error);
    return 0;
  }
  return cgbn::core_singleton_t<cgbn_env_t, LIMBS>::rem_ui32(a._limbs, div);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ bool cgbn_env_t<context_t, bits, syncable>::equals_ui32(const cgbn_t &a, const uint32_t value) const {
  return cgbn::core_t<cgbn_env_t>::equals_ui32(a._limbs, value);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ int32_t cgbn_env_t<context_t, bits, syncable>::compare_ui32(const cgbn_t &a, const uint32_t value) const {
  return cgbn::core_t<cgbn_env_t>::compare_ui32(a._limbs, value);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::extract_bits_ui32(const cgbn_t &a, const uint32_t start, const uint32_t len) const {
  return cgbn::core_t<cgbn_env_t>::extract_bits_ui32(a._limbs, start, len);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::insert_bits_ui32(cgbn_t &r, const cgbn_t &a, const uint32_t start, const uint32_t len, const uint32_t value) const {
  cgbn::core_t<cgbn_env_t>::insert_bits_ui32(r._limbs, a._limbs, start, len, value);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::binary_inverse_ui32(const uint32_t x) const {
  if(_context.check_errors()) {
    if((x & 0x01)==0) {
      _context.report_error(cgbn_inverse_does_not_exist_error);
      return 0;
    }
  }
  return cgbn::ubinary_inverse(x);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::gcd_ui32(const cgbn_t &a, const uint32_t value) const {
  if(value==0)
    return 0;
  return cgbn::ugcd(value, rem_ui32(a, value));
}


/* bn arithmetic routines */
template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ int32_t cgbn_env_t<context_t, bits, syncable>::add(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const {
  return cgbn::core_t<cgbn_env_t>::add(r._limbs, a._limbs, b._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ int32_t cgbn_env_t<context_t, bits, syncable>::sub(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const {
  return cgbn::core_t<cgbn_env_t>::sub(r._limbs, a._limbs, b._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ int32_t cgbn_env_t<context_t, bits, syncable>::negate(cgbn_t &r, const cgbn_t &a) const {
  return cgbn::core_t<cgbn_env_t>::negate(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::mul(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const {
  uint32_t add[LIMBS];
  
  cgbn::mpzero<LIMBS>(add);
  cgbn::core_singleton_t<cgbn_env_t, LIMBS>::mul(r._limbs, a._limbs, b._limbs, add);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::mul_high(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const {
  uint32_t add[LIMBS];
  
  cgbn::mpzero<LIMBS>(add);
  cgbn::core_singleton_t<cgbn_env_t, LIMBS>::mul_high(r._limbs, a._limbs, b._limbs, add);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::sqr(cgbn_t &r, const cgbn_t &a) const {
  uint32_t add[LIMBS];
  
  cgbn::mpzero<LIMBS>(add);
  cgbn::core_singleton_t<cgbn_env_t, LIMBS>::mul(r._limbs, a._limbs, a._limbs, add);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::sqr_high(cgbn_t &r, const cgbn_t &a) const {
  uint32_t add[LIMBS];
  
  cgbn::mpzero<LIMBS>(add);
  cgbn::core_singleton_t<cgbn_env_t, LIMBS>::mul_high(r._limbs, a._limbs, a._limbs, add);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::div(cgbn_t &q, const cgbn_t &num, const cgbn_t &denom) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;
  
  if(_context.check_errors()) {
    if(equals_ui32(denom, 0)) {
      _context.report_error(cgbn_division_by_zero_error);
      return;
    }
  }
  
  // division of padded values is the same as division of unpadded valuess
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_low, num._limbs, shift);
  core::bitwise_mask_and(num_high, num_low, shift);
  numthreads=TPI-core::clzt(num_high);
  singleton::div_wide(q._limbs, num_low, num_high, denom_local, numthreads);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::rem(cgbn_t &r, const cgbn_t &num, const cgbn_t &denom) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;

  if(_context.check_errors()) {
    if(equals_ui32(denom, 0)) {
      _context.report_error(cgbn_division_by_zero_error);
      return;
    }
  }
  // division of padded values is the same as division of unpadded valuess
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_low, num._limbs, shift);
  core::bitwise_mask_and(num_high, num_low, shift);
  core::bitwise_xor(num_low, num_low, num_high);
  numthreads=TPI-core::clzt(num_high);
  singleton::rem_wide(r._limbs, num_low, num_high, denom_local, numthreads);
  core::rotate_right(r._limbs, r._limbs, shift);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::div_rem(cgbn_t &q, cgbn_t &r, const cgbn_t &num, const cgbn_t &denom) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;

  if(_context.check_errors()) {
    if(equals_ui32(denom, 0)) {
      _context.report_error(cgbn_division_by_zero_error);
      return;
    }
  }
  // division of padded values is the same as division of unpadded valuess
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_low, num._limbs, shift);
  core::bitwise_mask_and(num_high, num_low, shift);
  core::bitwise_xor(num_low, num_low, num_high);
  numthreads=TPI-core::clzt(num_high);
  singleton::div_rem_wide(q._limbs, r._limbs, num_low, num_high, denom_local, numthreads);
  core::rotate_right(r._limbs, r._limbs, shift);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::sqrt(cgbn_t &s, const cgbn_t &a) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t shift, numthreads;
  uint32_t shifted[LIMBS];
  
  shift=core::clz(a._limbs);
  if(shift==UNPADDED_BITS) {
    cgbn::mpzero<LIMBS>(s._limbs);
    return;
  }
  numthreads=(UNPADDED_BITS+LIMBS*64-1-shift) / (LIMBS*64);
  core::rotate_left(shifted, a._limbs, shift & 0xFFFFFFFE);
  singleton::sqrt(s._limbs, shifted, numthreads);
  shift=(shift>>1) % (LIMBS*32);
  core::shift_right(s._limbs, s._limbs, shift);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::sqrt_rem(cgbn_t &s, cgbn_t &r, const cgbn_t &a) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t shift, numthreads;
  uint32_t remainder[LIMBS], temp[LIMBS];
  
  shift=core::clz(a._limbs);
  if(shift==UNPADDED_BITS) {
    cgbn::mpzero<LIMBS>(s._limbs);
    cgbn::mpzero<LIMBS>(r._limbs);
    return;
  }
  numthreads=(UNPADDED_BITS+LIMBS*64-1-shift) / (LIMBS*64);
  core::rotate_left(temp, a._limbs, shift & 0xFFFFFFFE);
  singleton::sqrt_rem(s._limbs, remainder, temp, numthreads);
  shift=(shift>>1) % (LIMBS*32);
  singleton::sqrt_resolve_rem(r._limbs, s._limbs, 0, remainder, shift);
  core::shift_right(s._limbs, s._limbs, shift);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ bool cgbn_env_t<context_t, bits, syncable>::equals(const cgbn_t &a, const cgbn_t &b) const {
  return cgbn::core_t<cgbn_env_t>::equals(a._limbs, b._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ int32_t cgbn_env_t<context_t, bits, syncable>::compare(const cgbn_t &a, const cgbn_t &b) const {
  return cgbn::core_t<cgbn_env_t>::compare(a._limbs, b._limbs);
}

/* wide arithmetic routines */
template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::mul_wide(cgbn_wide_t &r, const cgbn_t &a, const cgbn_t &b) const {
  uint32_t add[LIMBS];
  
  cgbn::mpzero<LIMBS>(add);
  cgbn::core_singleton_t<cgbn_env_t, LIMBS>::mul_wide(r._low._limbs, r._high._limbs, a._limbs, b._limbs, add);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::sqr_wide(cgbn_wide_t &r, const cgbn_t &a) const {
  uint32_t add[LIMBS];
  
  cgbn::mpzero<LIMBS>(add);
  cgbn::core_singleton_t<cgbn_env_t, LIMBS>::mul_wide(r._low._limbs, r._high._limbs, a._limbs, a._limbs, add);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::div_wide(cgbn_t &q, const cgbn_wide_t &num, const cgbn_t &denom) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;

  if(_context.check_errors()) {
    if(core::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(cgbn_division_overflow_error);
      return;
    }
  }
  
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_high, num._high._limbs, shift-(UNPADDED_BITS-BITS));
  core::rotate_left(num_low, num._low._limbs, shift);
  core::bitwise_mask_select(num_high, num_high, num_low, shift-(UNPADDED_BITS-BITS));
  numthreads=TPI-core::clzt(num_high);
  singleton::div_wide(q._limbs, num_low, num_high, denom_local, numthreads);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::rem_wide(cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;

  if(_context.check_errors()) {
    if(core::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(cgbn_division_overflow_error);
      return;
    }
  }
  
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_high, num._high._limbs, shift-(UNPADDED_BITS-BITS));
  core::rotate_left(num_low, num._low._limbs, shift);
  core::bitwise_mask_select(num_high, num_high, num_low, shift-(UNPADDED_BITS-BITS));
  core::bitwise_mask_and(num_low, num_low, shift-UNPADDED_BITS);
  numthreads=TPI-core::clzt(num_high);
  singleton::rem_wide(r._limbs, num_low, num_high, denom_local, numthreads);
  core::rotate_right(r._limbs, r._limbs, shift);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::div_rem_wide(cgbn_t &q, cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;

  if(_context.check_errors()) {
    if(core::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(cgbn_division_overflow_error);
      return;
    }
  }
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_high, num._high._limbs, shift-(UNPADDED_BITS-BITS));
  core::rotate_left(num_low, num._low._limbs, shift);
  core::bitwise_mask_select(num_high, num_high, num_low, shift-(UNPADDED_BITS-BITS));
  core::bitwise_mask_and(num_low, num_low, shift-UNPADDED_BITS);
  numthreads=TPI-core::clzt(num_high);
  singleton::div_rem_wide(q._limbs, r._limbs, num_low, num_high, denom_local, numthreads);
  core::rotate_right(r._limbs, r._limbs, shift);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::sqrt_wide(cgbn_t &s, const cgbn_wide_t &a) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t clz_shift, shift, numthreads;
  uint32_t high_shifted[LIMBS], low_shifted[LIMBS];
  
  clz_shift=core::clz(a._high._limbs);
  if(clz_shift==UNPADDED_BITS) {
    clz_shift=core::clz(a._low._limbs);
    if(clz_shift==UNPADDED_BITS) {
      cgbn::mpzero<LIMBS>(s._limbs);
      return;
    }
    clz_shift=clz_shift & 0xFFFFFFFE;
    cgbn::mpset<LIMBS>(high_shifted, a._low._limbs);
    cgbn::mpzero<LIMBS>(low_shifted);
    shift=clz_shift + UNPADDED_BITS;
  }
  else {
    clz_shift=clz_shift & 0xFFFFFFFE;
    cgbn::mpset<LIMBS>(high_shifted, a._high._limbs);
    core::rotate_left(low_shifted, a._low._limbs, clz_shift+(UNPADDED_BITS-BITS));
    shift=clz_shift+UNPADDED_BITS-BITS;
  }
  numthreads=(2*UNPADDED_BITS+LIMBS*64-1-shift) / (LIMBS*64);

  core::rotate_left(high_shifted, high_shifted, clz_shift);
  if(shift<2*UNPADDED_BITS-BITS) {
    core::bitwise_mask_select(high_shifted, high_shifted, low_shifted, clz_shift);
    core::bitwise_mask_and(low_shifted, low_shifted, (int32_t)(shift-UNPADDED_BITS));
  }

  singleton::sqrt_wide(s._limbs, low_shifted, high_shifted, numthreads);

  shift=(shift>>1) % (LIMBS*32);
  core::shift_right(s._limbs, s._limbs, shift);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::sqrt_rem_wide(cgbn_t &s, cgbn_wide_t &r, const cgbn_wide_t &a) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core_unpadded;
  typedef cgbn::core_t<cgbn_env_t> core_padded;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t group_thread=threadIdx.x & TPI-1;
  uint32_t clz_shift, shift, numthreads, c;
  uint32_t remainder[LIMBS], high_shifted[LIMBS], low_shifted[LIMBS];

  clz_shift=core_unpadded::clz(a._high._limbs);
  if(clz_shift==UNPADDED_BITS) {
    clz_shift=core_unpadded::clz(a._low._limbs);
    if(clz_shift==UNPADDED_BITS) {
      cgbn::mpzero<LIMBS>(s._limbs);
      cgbn::mpzero<LIMBS>(r._low._limbs);
      cgbn::mpzero<LIMBS>(r._high._limbs);
      return;
    }
    clz_shift=clz_shift & 0xFFFFFFFE;
    cgbn::mpset<LIMBS>(high_shifted, a._low._limbs);
    cgbn::mpzero<LIMBS>(low_shifted);
    shift=clz_shift + UNPADDED_BITS;
  }
  else {
    clz_shift=clz_shift & 0xFFFFFFFE;
    cgbn::mpset<LIMBS>(high_shifted, a._high._limbs);
    core_unpadded::rotate_left(low_shifted, a._low._limbs, clz_shift+(UNPADDED_BITS-BITS));
    shift=clz_shift+UNPADDED_BITS-BITS;
  }
  numthreads=(2*UNPADDED_BITS+LIMBS*64-1-shift) / (LIMBS*64);

  core_unpadded::rotate_left(high_shifted, high_shifted, clz_shift);
  if(shift<2*UNPADDED_BITS-BITS) {
    core_unpadded::bitwise_mask_select(high_shifted, high_shifted, low_shifted, clz_shift);
    core_unpadded::bitwise_mask_and(low_shifted, low_shifted, (int32_t)(shift-UNPADDED_BITS));
  }

  c=singleton::sqrt_rem_wide(s._limbs, remainder, low_shifted, high_shifted, numthreads);

  shift=(shift>>1) % (LIMBS*32);
  if(shift==0) {
    if(UNPADDED_BITS!=BITS)
      c=core_padded::clear_carry(remainder);
    cgbn::mpset<LIMBS>(r._low._limbs, remainder);
    cgbn::mpzero<LIMBS>(r._high._limbs);
    r._high._limbs[0]=(group_thread==0) ? c : 0;
  }
  else {
    singleton::sqrt_resolve_rem(r._low._limbs, s._limbs, c, remainder, shift);
    cgbn::mpzero<LIMBS>(r._high._limbs);
    if(UNPADDED_BITS!=BITS) {
      c=core_padded::clear_carry(r._low._limbs);
      r._high._limbs[0]=(group_thread==0) ? c : 0;
    }
    core_unpadded::shift_right(s._limbs, s._limbs, shift);
  }
}

/* bit counting */
template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::pop_count(const cgbn_t &a) const {
  return cgbn::core_t<cgbn_env_t>::pop_count(a._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::clz(const cgbn_t &a) const {
  return cgbn::core_t<cgbn_env_t>::clz(a._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::ctz(const cgbn_t &a) const {
  return cgbn::core_t<cgbn_env_t>::ctz(a._limbs);
}


/* logical, shifting, masking */
template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::bitwise_complement(cgbn_t &r, const cgbn_t &a) const {
  cgbn::core_t<cgbn_env_t>::bitwise_complement(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::bitwise_and(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const {
  cgbn::core_t<cgbn_env_t>::bitwise_and(r._limbs, a._limbs, b._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::bitwise_ior(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const {
  cgbn::core_t<cgbn_env_t>::bitwise_ior(r._limbs, a._limbs, b._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::bitwise_xor(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const {
  cgbn::core_t<cgbn_env_t>::bitwise_xor(r._limbs, a._limbs, b._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::bitwise_select(cgbn_t &r, const cgbn_t &clear, const cgbn_t &set, const cgbn_t &select) const {
  cgbn::core_t<cgbn_env_t>::bitwise_select(r._limbs, clear._limbs, set._limbs, select._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::bitwise_mask_copy(cgbn_t &r, const int32_t numbits) const {
  cgbn::core_t<cgbn_env_t>::bitwise_mask_copy(r._limbs, numbits);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::bitwise_mask_and(cgbn_t &r, const cgbn_t &a, const int32_t numbits) const {
  cgbn::core_t<cgbn_env_t>::bitwise_mask_and(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::bitwise_mask_ior(cgbn_t &r, const cgbn_t &a, const int32_t numbits) const {
  cgbn::core_t<cgbn_env_t>::bitwise_mask_ior(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::bitwise_mask_xor(cgbn_t &r, const cgbn_t &a, const int32_t numbits) const {
  cgbn::core_t<cgbn_env_t>::bitwise_mask_xor(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::bitwise_mask_select(cgbn_t &r, const cgbn_t &clear, const cgbn_t &set, const int32_t numbits) const {
  cgbn::core_t<cgbn_env_t>::bitwise_mask_select(r._limbs, clear._limbs, set._limbs, numbits);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::shift_left(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const {
  cgbn::core_t<cgbn_env_t>::shift_left(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::shift_right(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const {
  cgbn::core_t<cgbn_env_t>::shift_right(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::rotate_left(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const {
  cgbn::core_t<cgbn_env_t>::rotate_left(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::rotate_right(cgbn_t &r, const cgbn_t &a, const uint32_t numbits) const {
  cgbn::core_t<cgbn_env_t>::rotate_right(r._limbs, a._limbs, numbits);
}

#if 0
template<class context_t, uint32_t bits, cgbn_syncable_t syncable> template<uint32_t numbits>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::shift_left(cgbn_t &r, const cgbn_t &a) const {
  fwshift_left_constant<LIMBS, numbits>(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable> template<uint32_t numbits>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::shift_right(cgbn_t &r, const cgbn_t &a) const {
  fwshift_right_constant<LIMBS, numbits>(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable> template<uint32_t numbits>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::rotate_left(cgbn_t &r, const cgbn_t &a) const {
  fwrotate_left_constant<LIMBS, numbits>(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable> template<uint32_t numbits>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::rotate_right(cgbn_t &r, const cgbn_t &a) const {
  fwrotate_right_constant<LIMBS, numbits>(r._limbs, a._limbs);
}
#endif

/* accumulator APIs */

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ cgbn_env_t<context_t, bits, syncable>::cgbn_accumulator_t::cgbn_accumulator_t() {
  _carry=0;
  cgbn::mpzero<LIMBS>(_limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ int32_t cgbn_env_t<context_t, bits, syncable>::resolve(cgbn_t &sum, const cgbn_accumulator_t &accumulator) const {
  typedef cgbn::core_t<cgbn_env_t> core;

  uint32_t carry=accumulator._carry;
  int32_t  result;

  cgbn::mpset<LIMBS>(sum._limbs, accumulator._limbs);
  result=core::resolve_add(carry, sum._limbs);
  core::clear_padding(sum._limbs);
  return result;
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::set_ui32(cgbn_accumulator_t &accumulator, const uint32_t value) const {
  uint32_t group_thread=threadIdx.x & TPI-1;

  accumulator._carry=0;
  accumulator._limbs[0]=(group_thread==0) ? value : 0;
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    accumulator._limbs[index]=0;
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::add_ui32(cgbn_accumulator_t &accumulator, const uint32_t value) const {
  uint32_t group_thread=threadIdx.x & TPI-1;

  cgbn::chain_t<> chain;
  accumulator._limbs[0]=chain.add(accumulator._limbs[0], (group_thread==0) ? value : 0);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    accumulator._limbs[index]=chain.add(accumulator._limbs[index], 0);
  accumulator._carry=chain.add(accumulator._carry, 0);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::sub_ui32(cgbn_accumulator_t &accumulator, const uint32_t value) const {
  uint32_t group_thread=threadIdx.x & TPI-1;

  cgbn::chain_t<> chain;
  chain.sub(0, group_thread);
  accumulator._limbs[0]=chain.sub(accumulator._limbs[0], (group_thread==0) ? value : 0);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    accumulator._limbs[index]=chain.sub(accumulator._limbs[index], 0);
    
  if(PADDING==0)
    accumulator._carry=chain.add(accumulator._carry, (group_thread==TPI-1) ? 0xFFFFFFFF : 0);
  else
    accumulator._carry=chain.add(accumulator._carry, 0);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::set(cgbn_accumulator_t &accumulator, const cgbn_t &value) const {
  accumulator._carry=0;
  cgbn::mpset<LIMBS>(accumulator._limbs, value._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::add(cgbn_accumulator_t &accumulator, const cgbn_t &value) const {
  cgbn::chain_t<> chain;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    accumulator._limbs[index]=chain.add(accumulator._limbs[index], value._limbs[index]);
  accumulator._carry=chain.add(accumulator._carry, 0);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::sub(cgbn_accumulator_t &accumulator, const cgbn_t &value) const {
  uint32_t group_thread=threadIdx.x & TPI-1;

  cgbn::chain_t<> chain;
  chain.sub(0, group_thread);
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    accumulator._limbs[index]=chain.sub(accumulator._limbs[index], value._limbs[index]);

  if(PADDING==0)
    accumulator._carry=chain.add(accumulator._carry, (group_thread==TPI-1) ? 0xFFFFFFFF : 0);
  else
    accumulator._carry=chain.add(accumulator._carry, 0);
}

/* math */
template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::binary_inverse(cgbn_t &r, const cgbn_t &x) const {
  uint32_t low;
  
  if(_context.check_errors()) {
    low=cgbn::core_t<cgbn_env_t>::get_ui32(x._limbs);
    if((low & 0x01)==0) {
      _context.report_error(cgbn_inverse_does_not_exist_error);
      return;
    }
  }

  cgbn::core_t<cgbn_env_t>::binary_inverse(r._limbs, x._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ bool cgbn_env_t<context_t, bits, syncable>::modular_inverse(cgbn_t &r, const cgbn_t &x, const cgbn_t &m) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  
  return cgbn::core_t<unpadded>::modular_inverse(r._limbs, x._limbs, m._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::modular_power(cgbn_t &r, const cgbn_t &a, const cgbn_t &k, const cgbn_t &m) const {
  cgbn_wide_t wide;
  cgbn_t      current, square, approx;
  int32_t     bit, m_clz, last;

  // FIX FIX FIX -- errors get checked again and again
  
  if(_context.check_errors()) {
    if(compare(a, m)>=0) {
      _context.report_error(cgbn_division_overflow_error);
      return;
    }
  }

  set_ui32(current, 1);
  set(square, a);
  m_clz=barrett_approximation(approx, m);
  last=bits-1-clz(k);
  if(last==-1) {
    set_ui32(r, 1);
    return;
  }
  for(bit=0;bit<last;bit++) {
    if(extract_bits_ui32(k, bit, 1)==1) {
      mul_wide(wide, current, square);
      barrett_rem_wide(current, wide, m, approx, m_clz);
    }
    mul_wide(wide, square, square);
    barrett_rem_wide(square, wide, m, approx, m_clz);
  }
  mul_wide(wide, current, square);
  barrett_rem_wide(r, wide, m, approx, m_clz);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::gcd(cgbn_t &r, const cgbn_t &a, const cgbn_t &b) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  
  cgbn::core_t<unpadded>::gcd(r._limbs, a._limbs, b._limbs);
}

/* fast division: common divisor / modulus */
template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::bn2mont(cgbn_t &mont, const cgbn_t &bn, const cgbn_t &n) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t num_low[LIMBS], num_high[LIMBS], n_local[LIMBS];
  uint32_t shift, low;

  low=core::get_ui32(n._limbs);
  
  if(_context.check_errors()) {
    if((low & 0x01)==0) {
      _context.report_error(cgbn_modulus_not_odd_error);
      return 0;
    }
    if(compare(bn, n)>=0) {
      _context.report_error(cgbn_division_overflow_error);
      return 0;
    }
  }

  // for padded values, we use a larger R
  cgbn::mpzero<LIMBS>(num_low);
  shift=core::clz(n._limbs);
  core::rotate_left(n_local, n._limbs, shift);
  core::rotate_left(num_high, bn._limbs, shift);
  singleton::rem_wide(mont._limbs, num_low, num_high, n_local, TPI);
  core::shift_right(mont._limbs, mont._limbs, shift);
  return -cgbn::ubinary_inverse(low);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::mont2bn(cgbn_t &bn, const cgbn_t &mont, const cgbn_t &n, const uint32_t np0) const {
  uint32_t zeros[LIMBS];

  cgbn::mpzero<LIMBS>(zeros);

  // mont_reduce_wide returns 0<=res<=n
  cgbn::core_singleton_t<cgbn_env_t, LIMBS>::mont_reduce_wide(bn._limbs, mont._limbs, zeros, n._limbs, np0, true);

  // handle the case of res==n
  if(cgbn::core_t<cgbn_env_t>::equals(bn._limbs, n._limbs))  
    cgbn::mpzero<LIMBS>(bn._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::mont_mul(cgbn_t &r, const cgbn_t &a, const cgbn_t &b, const cgbn_t &n, const uint32_t np0) const {
  cgbn::core_singleton_t<cgbn_env_t, LIMBS>::mont_mul(r._limbs, a._limbs, b._limbs, n._limbs, np0);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::mont_sqr(cgbn_t &r, const cgbn_t &a, const cgbn_t &n, const uint32_t np0) const {
  cgbn::core_singleton_t<cgbn_env_t, LIMBS>::mont_mul(r._limbs, a._limbs, a._limbs, n._limbs, np0);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::mont_reduce_wide(cgbn_t &r, const cgbn_wide_t &a, const cgbn_t &n, const uint32_t np0) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t low[LIMBS], high[LIMBS];
  
  cgbn::mpset<LIMBS>(low, a._low._limbs);
  cgbn::mpset<LIMBS>(high, a._high._limbs);
  
  if(PADDING!=0) {
    core::rotate_right(high, high, UNPADDED_BITS-BITS);
    core::bitwise_mask_select(low, high, low, BITS);
    core::bitwise_mask_and(high, high, BITS);
  }
  
  // mont_reduce_wide returns 0<=res<=n
  singleton::mont_reduce_wide(r._limbs, low, high, n._limbs, np0, false);

  // handle the case of res==n
  if(core::equals(r._limbs, n._limbs))  
    cgbn::mpzero<LIMBS>(r._limbs);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ uint32_t cgbn_env_t<context_t, bits, syncable>::barrett_approximation(cgbn_t &approx, const cgbn_t &denom) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t shift, shifted[LIMBS], low[LIMBS], high[LIMBS];

  shift=core::clz(denom._limbs);
  if(_context.check_errors()) {
    if(shift==UNPADDED_BITS) {
      _context.report_error(cgbn_division_by_zero_error);
      return 0xFFFFFFFF;
    }
  }

  if(shift==UNPADDED_BITS)
    return 0xFFFFFFFF;

  core::rotate_left(shifted, denom._limbs, shift);
  
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    low[index]=0xFFFFFFFF;
    high[index]=~shifted[index];  // high=0xFFFFFFFF - shifted[index]
  }
  
  singleton::div_wide(approx._limbs, low, high, shifted, TPI);
  return shift;
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::barrett_div(cgbn_t &q, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t sync, group_thread=threadIdx.x & TPI-1;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c, sub=0;

  sync=core::sync_mask();
  core::shift_right(high, num._limbs, UNPADDED_BITS-denom_clz);
  cgbn::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      quotient[index]=0xFFFFFFFF;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);
  
  word=-__shfl_sync(sync, high[0], 0, TPI);
  c=cgbn::mpsub<LIMBS>(low, num._limbs, low);
  word-=core::fast_propagate_sub(c, low);
  while(word!=0) {
    sub++;
    c=cgbn::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core::fast_propagate_add(c, low);
  }
  core::sub_ui32(q._limbs, quotient, sub);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::barrett_rem(cgbn_t &r, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t sync, group_thread=threadIdx.x & TPI-1;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c;

  sync=core::sync_mask();
  core::shift_right(high, num._limbs, UNPADDED_BITS-denom_clz);
  cgbn::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      quotient[index]=0xFFFFFFFF;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);

  word=-__shfl_sync(sync, high[0], 0, TPI);
  c=cgbn::mpsub<LIMBS>(low, num._limbs, low);
  word-=core::fast_propagate_sub(c, low);
  while(word!=0) {
    c=cgbn::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core::fast_propagate_add(c, low);
  }
  cgbn::mpset<LIMBS>(r._limbs, low);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::barrett_div_rem(cgbn_t &q, cgbn_t &r, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t sync, group_thread=threadIdx.x & TPI-1;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c, sub=0;

  sync=core::sync_mask();
  core::shift_right(high, num._limbs, UNPADDED_BITS-denom_clz);
  cgbn::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      quotient[index]=0xFFFFFFFF;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);

  word=-__shfl_sync(sync, high[0], 0, TPI);
  c=cgbn::mpsub<LIMBS>(low, num._limbs, low);
  word-=core::fast_propagate_sub(c, low);
  while(word!=0) {
    sub++;
    c=cgbn::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core::fast_propagate_add(c, low);
  }
  core::sub_ui32(q._limbs, quotient, sub);
  cgbn::mpset<LIMBS>(r._limbs, low);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::barrett_div_wide(cgbn_t &q, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core_unpadded;
  typedef cgbn::core_t<cgbn_env_t> core_padded;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t sync, group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c, sub=0;

  if(_context.check_errors()) {
    if(core_unpadded::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(cgbn_division_overflow_error);
      return;
    }
  }

  sync=core_unpadded::sync_mask();
  word=__shfl_sync(sync, num._high._limbs[0], 0, TPI);
  core_unpadded::rotate_left(low, num._low._limbs, denom_clz);
  core_unpadded::rotate_left(high, num._high._limbs, denom_clz-(UNPADDED_BITS-BITS));
  core_unpadded::bitwise_mask_select(high, high, low, denom_clz-(UNPADDED_BITS-BITS));
  cgbn::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core_padded::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      if(PADDING==0)
        quotient[index]=0xFFFFFFFF;
      else
        quotient[index]=(group_base<BITS/32-index) ? 0xFFFFFFFF : 0;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);

  if(PADDING==0)
    word=word-__shfl_sync(sync, high[0], 0, TPI);
  else {
    word=word-__shfl_sync(sync, low[PAD_LIMB], PAD_THREAD, TPI);
    core_padded::clear_padding(low);
  }
    
  c=cgbn::mpsub<LIMBS>(low, num._low._limbs, low);
  word-=core_padded::fast_propagate_sub(c, low);
  while(word!=0) {
    sub++;
    c=cgbn::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core_padded::fast_propagate_add(c, low);
  }
  core_unpadded::sub_ui32(q._limbs, quotient, sub);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::barrett_rem_wide(cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core_unpadded;
  typedef cgbn::core_t<cgbn_env_t> core_padded;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
    
  uint32_t sync, group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c;

  if(_context.check_errors()) {
    if(core_unpadded::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(cgbn_division_overflow_error);
      return;
    }
  }

  sync=core_unpadded::sync_mask();
  word=__shfl_sync(sync, num._high._limbs[0], 0, TPI);
  core_unpadded::rotate_left(low, num._low._limbs, denom_clz);
  core_unpadded::rotate_left(high, num._high._limbs, denom_clz-(UNPADDED_BITS-BITS));
  core_unpadded::bitwise_mask_select(high, high, low, denom_clz-(UNPADDED_BITS-BITS));
  cgbn::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core_padded::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      if(PADDING==0)
        quotient[index]=0xFFFFFFFF;
      else
        quotient[index]=(group_base<BITS/32-index) ? 0xFFFFFFFF : 0;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);

  if(PADDING==0)
    word=word-__shfl_sync(sync, high[0], 0, TPI);
  else {
    word=word-__shfl_sync(sync, low[PAD_LIMB], PAD_THREAD, TPI);
    core_padded::clear_padding(low);
  }
    
  c=cgbn::mpsub<LIMBS>(low, num._low._limbs, low);
  word-=core_padded::fast_propagate_sub(c, low);
  while(word!=0) {
    c=cgbn::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core_padded::fast_propagate_add(c, low);
  }
  cgbn::mpset<LIMBS>(r._limbs, low);
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::barrett_div_rem_wide(cgbn_t &q, cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz) const {
  typedef cgbn::unpadded_t<cgbn_env_t> unpadded;
  typedef cgbn::core_t<unpadded> core_unpadded;
  typedef cgbn::core_t<cgbn_env_t> core_padded;
  typedef cgbn::core_singleton_t<unpadded, LIMBS> singleton;
    
  uint32_t sync, group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c, sub=0;

  if(_context.check_errors()) {
    if(core_unpadded::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(cgbn_division_overflow_error);
      return;
    }
  }

  sync=core_unpadded::sync_mask();
  word=__shfl_sync(sync, num._high._limbs[0], 0, TPI);
  core_unpadded::rotate_left(low, num._low._limbs, denom_clz);
  core_unpadded::rotate_left(high, num._high._limbs, denom_clz-(UNPADDED_BITS-BITS));
  core_unpadded::bitwise_mask_select(high, high, low, denom_clz-(UNPADDED_BITS-BITS));
  cgbn::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core_padded::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      if(PADDING==0)
        quotient[index]=0xFFFFFFFF;
      else
        quotient[index]=(group_base<BITS/32-index) ? 0xFFFFFFFF : 0;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);

  if(PADDING==0)
    word=word-__shfl_sync(sync, high[0], 0, TPI);
  else {
    word=word-__shfl_sync(sync, low[PAD_LIMB], PAD_THREAD, TPI);
    core_padded::clear_padding(low);
  }
    
  c=cgbn::mpsub<LIMBS>(low, num._low._limbs, low);
  word-=core_padded::fast_propagate_sub(c, low);
  while(word!=0) {
    sub++;
    c=cgbn::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core_padded::fast_propagate_add(c, low);
  }
  core_unpadded::sub_ui32(q._limbs, quotient, sub);
  cgbn::mpset<LIMBS>(r._limbs, low);
}

/* load/store routines */
template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::load(cgbn_t &r, cgbn_mem_t<bits> *const address) const {
  int32_t group_thread=threadIdx.x & TPI-1;
  int32_t limb;

  #pragma unroll
  for(limb=0;limb<LIMBS;limb++) {
    if(PADDING!=0) {
      r._limbs[limb]=0;
      if(group_thread*LIMBS<BITS/32-limb) 
        r._limbs[limb]=address->_limbs[group_thread*LIMBS + limb];
    }
    else
      r._limbs[limb]=address->_limbs[group_thread*LIMBS + limb];
  }
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::store(cgbn_mem_t<bits> *address, const cgbn_t &a) const {
  int32_t group_thread=threadIdx.x & TPI-1;
  int32_t limb;

  #pragma unroll
  for(limb=0;limb<LIMBS;limb++) {
    if(PADDING!=0) {
      if(group_thread*LIMBS<BITS/32-limb)
        address->_limbs[group_thread*LIMBS + limb]=a._limbs[limb];
#if 1
      else
        if(a._limbs[limb]!=0) {
          printf("BAD LIMB: %d %d %d\n", blockIdx.x, threadIdx.x, limb);
          __trap();
        }
#endif
    }
    else
      address->_limbs[group_thread*LIMBS + limb]=a._limbs[limb];
  }
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::load(cgbn_t &r, cgbn_local_t *const address) const {
  int32_t limb;

  #pragma unroll
  for(limb=0;limb<LIMBS;limb++)
    r._limbs[limb]=address->_limbs[limb];
}

template<class context_t, uint32_t bits, cgbn_syncable_t syncable>
__device__ __forceinline__ void cgbn_env_t<context_t, bits, syncable>::store(cgbn_local_t *address, const cgbn_t &a) const {
  int32_t limb;

  #pragma unroll
  for(limb=0;limb<LIMBS;limb++)
    address->_limbs[limb]=a._limbs[limb];
}
