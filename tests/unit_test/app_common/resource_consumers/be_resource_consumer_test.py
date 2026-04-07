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

from nvflare.apis.resource_manager_spec import ResourceConsumerSpec
from nvflare.app_common.resource_consumers.be_resource_consumer import BEResourceConsumer


class TestBEResourceConsumerIsSpec:
    def test_is_subclass_of_resource_consumer_spec(self):
        assert issubclass(BEResourceConsumer, ResourceConsumerSpec)

    def test_instance_is_resource_consumer_spec(self):
        assert isinstance(BEResourceConsumer(), ResourceConsumerSpec)


class TestBEResourceConsumerConsume:
    def test_returns_none(self):
        assert BEResourceConsumer().consume({"num_of_gpus": 2}) is None

    def test_empty_resources_returns_none(self):
        assert BEResourceConsumer().consume({}) is None

    def test_does_not_raise(self):
        BEResourceConsumer().consume({"mem_per_gpu_in_GiB": 16, "num_of_gpus": 4})

    def test_large_resources_does_not_raise(self):
        BEResourceConsumer().consume({"num_of_gpus": 1000, "cpu_cores": 512, "mem_GiB": 2048})

    def test_non_standard_keys_does_not_raise(self):
        BEResourceConsumer().consume({"arbitrary_key": "value", "another": 42})

    def test_called_multiple_times_does_not_raise(self):
        consumer = BEResourceConsumer()
        for i in range(10):
            consumer.consume({"num_of_gpus": i})
