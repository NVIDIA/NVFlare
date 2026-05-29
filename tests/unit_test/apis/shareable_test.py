# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np

from nvflare.apis.shareable import Shareable, make_copy


class TensorLike:
    pass


class SlottedWrapper:
    __slots__ = ("nested", "tensor")

    def __init__(self, tensor):
        self.tensor = tensor
        self.nested = {"values": [1, 2, 3]}


def test_make_copy_deep_copies_containers_and_reuses_large_values():
    bytes_data = b"x" * 2048
    bytearray_data = bytearray(b"abc")
    memoryview_data = memoryview(bytearray(b"view"))
    ndarray_data = np.arange(4)
    tensor_data = TensorLike()
    slotted_wrapper = SlottedWrapper(tensor_data)
    header_array = np.arange(2)
    source = Shareable(
        {
            "payload": {
                "items": [
                    {
                        "bytes": bytes_data,
                        "bytearray": bytearray_data,
                        "memoryview": memoryview_data,
                        "ndarray": ndarray_data,
                        "tensor": tensor_data,
                        "slotted_wrapper": slotted_wrapper,
                    }
                ],
                "metrics": [1, 2, 3],
            }
        }
    )
    source.set_header("keep", {"nested": ["value"], "array": header_array})
    source.set_header("drop", {"nested": ["secret"]})

    copied = make_copy(source, exclude_headers=["drop"], no_copy_types=(TensorLike,))

    assert isinstance(copied, Shareable)
    assert copied is not source
    assert copied["payload"] is not source["payload"]
    assert copied["payload"]["items"] is not source["payload"]["items"]
    assert copied["payload"]["items"][0] is not source["payload"]["items"][0]
    assert copied["payload"]["items"][0]["bytes"] is bytes_data
    assert copied["payload"]["items"][0]["bytearray"] is bytearray_data
    assert copied["payload"]["items"][0]["memoryview"] is memoryview_data
    assert copied["payload"]["items"][0]["ndarray"] is ndarray_data
    assert copied["payload"]["items"][0]["tensor"] is tensor_data
    assert copied["payload"]["items"][0]["slotted_wrapper"] is not slotted_wrapper
    assert copied["payload"]["items"][0]["slotted_wrapper"].tensor is tensor_data
    assert copied["payload"]["items"][0]["slotted_wrapper"].nested is not slotted_wrapper.nested
    assert copied["payload"]["metrics"] is not source["payload"]["metrics"]
    assert copied.get_header("keep") is not source.get_header("keep")
    assert copied.get_header("keep")["nested"] == source.get_header("keep")["nested"]
    assert copied.get_header("keep")["nested"] is not source.get_header("keep")["nested"]
    assert copied.get_header("keep")["array"] is header_array
    assert copied.get_header("drop") is None
    assert source.get_header("drop") == {"nested": ["secret"]}

    copied["payload"]["metrics"].append(4)
    copied.get_header("keep")["nested"].append("copy")
    copied["payload"]["items"][0]["slotted_wrapper"].nested["values"].append(4)

    assert source["payload"]["metrics"] == [1, 2, 3]
    assert source.get_header("keep")["nested"] == ["value"]
    assert slotted_wrapper.nested["values"] == [1, 2, 3]


def test_make_copy_accepts_single_no_copy_type():
    tensor_data = TensorLike()
    source = Shareable({"tensor": tensor_data})

    copied = make_copy(source, no_copy_types=TensorLike)

    assert copied is not source
    assert copied["tensor"] is tensor_data
