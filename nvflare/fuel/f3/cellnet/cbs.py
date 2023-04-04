# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.fuel.f3.cellnet.cell import CellAgent
from nvflare.fuel.f3.message import Message


def cell_connected_cb_signature(connected_cell: CellAgent, *args, **kwargs):
    """
    This is the signature of the cell_connected callback.

    Args:
        connected_cell: the cell that just got connected
        *args:
        **kwargs:

    Returns:

    """
    pass


def cell_disconnected_cb_signature(disconnected_cell: CellAgent, *args, **kwargs):
    pass


def request_cb_signature(request: Message, *args, **kwargs) -> Message:
    pass


def message_interceptor_signature(message: Message, *args, **kwargs) -> Message:
    pass


def filter_cb_signature(message: Message, *args, **kwargs) -> Message:
    pass


def cleanup_cb_signature(*args, **kwargs):
    pass
