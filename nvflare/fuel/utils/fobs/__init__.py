# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.fuel.utils.fobs.decomposer import Decomposer
from nvflare.fuel.utils.fobs.fobs import (
    auto_register_enum_types,
    deserialize,
    deserialize_stream,
    get_dot_handler,
    num_decomposers,
    register,
    register_data_classes,
    register_enum_types,
    register_folder,
    reset,
    serialize,
    serialize_stream,
)
from nvflare.fuel.utils.fobs.lobs import (
    dump_to_bytes,
    dump_to_file,
    dump_to_stream,
    load_from_bytes,
    load_from_file,
    load_from_stream,
)

# aliases for compatibility to Pickle/json
load = load_from_stream
dump = dump_to_stream
loads = load_from_bytes
dumps = dump_to_bytes
loadf = load_from_file
dumpf = dump_to_file


class FOBSContextKey:
    CELL = "cell"
    CORE_CELL = "core_cell"
    MESSAGE = "message"
    ABORT_SIGNAL = "abort_signal"
    DOWNLOAD_REQ_TIMEOUT = "download_req_timeout"
    SEC_CREDS = "sec_creds"
    NUM_RECEIVERS = "num_receivers"
    # When True, ViaDownloaderDecomposer will NOT download tensors at this hop.
    # Instead it creates LazyDownloadRef placeholders that preserve the original
    # source FQCN/ref_id so the reference can be forwarded verbatim to the next
    # hop (e.g. a subprocess agent), which then downloads directly from the
    # originating source.  This eliminates intermediate tensor copies at the
    # forwarding node (the CJ) and is the foundation of the B1 pass-through
    # architecture.
    PASS_THROUGH = "pass_through"
    # Optional callable set by FlareAgent before serialising a result message
    # when reverse PASS_THROUGH is active (subprocess → CJ → server).  Signature:
    #   cb(tx_id: str, status: str, base_objs: list) -> None
    # _create_downloader() chains this callback into the transaction_done_cb so
    # it fires when the server (or Swarm peer) finishes downloading from this
    # subprocess's DownloadService.  FlareAgent waits on a threading.Event
    # backed by this callback to gate subprocess exit on download completion.
    DOWNLOAD_COMPLETE_CB = "download_complete_cb"
