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

from nvflare.apis.dxo import DXO, DataKind, from_shareable, get_leaf_dxos
from nvflare.app_common.ccwf.swarm_client_ctl import _materialize_shareable_for_aggregator


class _Aggregator:
    pass


class _LazyAwareAggregator:
    accepts_lazy_tensors = True


class _TempRef:
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


class _LazyRef:
    def __init__(self, value, temp_ref: _TempRef):
        self.value = value
        self._temp_ref = temp_ref
        self.resolve_calls = 0

    def resolve(self):
        self.resolve_calls += 1
        return self.value


class TestSwarmLazyCompatibility:
    def test_materialize_for_non_lazy_aware_aggregator(self):
        temp_ref = _TempRef()
        lazy_ref = _LazyRef(4.2, temp_ref)
        shareable = DXO(data_kind=DataKind.WEIGHTS, data={"w": lazy_ref}).to_shareable()

        changed = _materialize_shareable_for_aggregator(
            result=shareable, aggregator=_Aggregator(), stream_to_disk=True
        )

        dxo = from_shareable(shareable)
        assert changed is True
        assert dxo.data["w"] == 4.2
        assert lazy_ref.resolve_calls == 1
        assert temp_ref.cleaned is True

    def test_keep_lazy_for_lazy_aware_aggregator(self):
        temp_ref = _TempRef()
        lazy_ref = _LazyRef(5.5, temp_ref)
        shareable = DXO(data_kind=DataKind.WEIGHTS, data={"w": lazy_ref}).to_shareable()

        changed = _materialize_shareable_for_aggregator(
            result=shareable, aggregator=_LazyAwareAggregator(), stream_to_disk=True
        )

        dxo = from_shareable(shareable)
        assert changed is False
        assert dxo.data["w"] is lazy_ref
        assert lazy_ref.resolve_calls == 0
        assert temp_ref.cleaned is False

    def test_materialize_collection_dxo(self):
        temp_ref = _TempRef()
        lazy_ref = _LazyRef(9.0, temp_ref)
        collection = DXO(
            data_kind=DataKind.COLLECTION,
            data={
                "weights": DXO(data_kind=DataKind.WEIGHTS, data={"w": lazy_ref}),
                "metrics": DXO(data_kind=DataKind.METRICS, data={"acc": 1.0}),
            },
        )
        shareable = collection.to_shareable()

        changed = _materialize_shareable_for_aggregator(
            result=shareable, aggregator=_Aggregator(), stream_to_disk=True
        )

        dxo = from_shareable(shareable)
        leaves, errors = get_leaf_dxos(dxo)
        assert errors == []
        assert changed is True
        assert leaves[".weights"].data["w"] == 9.0
        assert lazy_ref.resolve_calls == 1
        assert temp_ref.cleaned is True

    def test_no_materialize_when_disk_streaming_disabled(self):
        temp_ref = _TempRef()
        lazy_ref = _LazyRef(1.5, temp_ref)
        shareable = DXO(data_kind=DataKind.WEIGHTS, data={"w": lazy_ref}).to_shareable()

        changed = _materialize_shareable_for_aggregator(
            result=shareable, aggregator=_Aggregator(), stream_to_disk=False
        )

        dxo = from_shareable(shareable)
        assert changed is False
        assert dxo.data["w"] is lazy_ref
        assert lazy_ref.resolve_calls == 0
        assert temp_ref.cleaned is False
