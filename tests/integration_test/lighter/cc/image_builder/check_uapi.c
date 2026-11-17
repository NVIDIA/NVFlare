/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Compile against the pinned kernel's UAPI headers on Linux x86-64.
 * These are the constants and layouts used by cvm/runtime/platforms.py.
 * This check does not invoke either ioctl or require a TEE device.
 */
#include <stddef.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <linux/tdx-guest.h>
#include <linux/sev-guest.h>

#if !defined(__linux__) || !defined(__x86_64__)
#error "The CVM local-report ABI is qualified only on Linux x86-64"
#endif

_Static_assert(TDX_CMD_GET_REPORT0 == 0xC4405401UL, "TDX ioctl changed");
_Static_assert(sizeof(struct tdx_report_req) == 1088, "TDX request size changed");
_Static_assert(offsetof(struct tdx_report_req, reportdata) == 0, "TDX nonce offset changed");
_Static_assert(offsetof(struct tdx_report_req, tdreport) == 64, "TDX report offset changed");
_Static_assert(TDX_REPORTDATA_LEN == 64 && TDX_REPORT_LEN == 1024, "TDX field sizes changed");
_Static_assert(SNP_GET_REPORT == 0xC0205300UL, "SNP ioctl changed");
_Static_assert(sizeof(struct snp_guest_request_ioctl) == 32, "SNP ioctl request size changed");
_Static_assert(offsetof(struct snp_guest_request_ioctl, msg_version) == 0, "SNP version offset changed");
_Static_assert(offsetof(struct snp_guest_request_ioctl, req_data) == 8, "SNP request offset changed");
_Static_assert(offsetof(struct snp_guest_request_ioctl, resp_data) == 16, "SNP response offset changed");
_Static_assert(offsetof(struct snp_guest_request_ioctl, exitinfo2) == 24, "SNP error offset changed");
_Static_assert(sizeof(struct snp_report_req) == 96, "SNP report request size changed");
_Static_assert(offsetof(struct snp_report_req, vmpl) == 64, "SNP VMPL offset changed");
_Static_assert(sizeof(struct snp_report_resp) == 4000, "SNP report response size changed");

int main(void)
{
    puts("PASS: TDX and SNP local-report ioctl constants and request layouts match kernel UAPI");
    return 0;
}
