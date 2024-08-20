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

cudaError_t cgbn_error_report_alloc(cgbn_error_report_t **report) {
  cudaError_t status;

  status=cudaMallocManaged((void **)report, sizeof(cgbn_error_report_t));
  if(status!=0)
    return status;
  (*report)->_error=cgbn_no_error;
  (*report)->_instance=0xFFFFFFFFu;
  (*report)->_threadIdx.x=0xFFFFFFFFu;
  (*report)->_threadIdx.y=0xFFFFFFFFu;
  (*report)->_threadIdx.z=0xFFFFFFFFu;
  (*report)->_blockIdx.x=0xFFFFFFFFu;
  (*report)->_blockIdx.y=0xFFFFFFFFu;
  (*report)->_blockIdx.z=0xFFFFFFFFu;
  return status;
}

cudaError_t cgbn_error_report_free(cgbn_error_report_t *report) {
  return cudaFree(report);
}

bool cgbn_error_report_check(cgbn_error_report_t *report) {
  return report->_error!=cgbn_no_error;
}

void cgbn_error_report_reset(cgbn_error_report_t *report) {
  report->_error=cgbn_no_error;
  report->_instance=0xFFFFFFFFu;
  report->_threadIdx.x=0xFFFFFFFFu;
  report->_threadIdx.y=0xFFFFFFFFu;
  report->_threadIdx.z=0xFFFFFFFFu;
  report->_blockIdx.x=0xFFFFFFFFu;
  report->_blockIdx.y=0xFFFFFFFFu;
  report->_blockIdx.z=0xFFFFFFFFu;
}

const char *cgbn_error_string(cgbn_error_report_t *report) {
  if(report->_error==cgbn_no_error)
    return NULL;
  switch(report->_error) {
    case cgbn_unsupported_threads_per_instance:
      return "unsupported threads per instance";
    case cgbn_unsupported_size:
      return "unsupported size";
    case cgbn_unsupported_limbs_per_thread:
      return "unsupported limbs per thread";
    case cgbn_unsupported_operation:
      return "unsupported operation";
    case cgbn_threads_per_block_mismatch:
      return "TPB does not match blockDim.x";
    case cgbn_threads_per_instance_mismatch:
      return "TPI does not match env_t::TPI";
    case cgbn_division_by_zero_error:
      return "division by zero";
    case cgbn_division_overflow_error:
      return "division overflow";
    case cgbn_invalid_montgomery_modulus_error:
      return "invalid montgomery modulus";
    case cgbn_modulus_not_odd_error:
      return "invalid modulus (it must be odd)";
    case cgbn_inverse_does_not_exist_error:
      return "inverse does not exist";
    case cgbn_no_error:
      return NULL;
  }
  return NULL;
}
