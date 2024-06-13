# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ctypes
import os
from contextlib import contextmanager

import numpy as np


def _check_call(rc: int) -> None:
    assert rc == 0


plugin_path = os.path.join(
    os.path.dirname(os.path.normpath(os.path.abspath(__file__))), os.pardir, "build", "libproc_nvflare.so"
)


@contextmanager
def load_plugin():
    nvflare = ctypes.cdll.LoadLibrary(plugin_path)
    nvflare.FederatedPluginCreate.restype = ctypes.c_void_p
    handle = ctypes.c_void_p(nvflare.FederatedPluginCreate(ctypes.c_int(0), None))
    try:
        yield nvflare, handle
    finally:
        _check_call(nvflare.FederatedPluginClose(handle))


def test_load():
    with load_plugin() as nvflare:
        pass


def test_grad():
    array = np.arange(16, dtype=np.float32)
    out = ctypes.POINTER(ctypes.c_uint8)()
    out_len = ctypes.c_size_t()

    with load_plugin() as (nvflare, handle):
        _check_call(
            nvflare.FederatedPluginEncryptGPairs(
                handle,
                array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                array.size,
                ctypes.byref(out),
                ctypes.byref(out_len),
            )
        )

        out1 = ctypes.POINTER(ctypes.c_uint8)()
        out_len1 = ctypes.c_size_t()

        _check_call(
            nvflare.FederatedPluginEncryptGPairs(
                handle,
                out,
                out_len,
                ctypes.byref(out1),
                ctypes.byref(out_len1),
            )
        )
