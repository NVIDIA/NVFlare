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
from nvflare.app_opt.xgboost.histogram_based_v2.sec.dam import DamDecoder, DamEncoder

DATA_SET = 123456
INT_ARRAY = [123, 456, 789]
FLOAT_ARRAY = [1.2, 2.3, 3.4, 4.5]


class TestDam:
    def test_encode_decode(self):
        encoder = DamEncoder(DATA_SET)
        encoder.add_int_array(INT_ARRAY)
        encoder.add_float_array(FLOAT_ARRAY)
        buffer = encoder.finish()

        decoder = DamDecoder(buffer)
        assert decoder.is_valid()
        assert decoder.get_data_set_id() == DATA_SET

        int_array = decoder.decode_int_array()
        assert int_array == INT_ARRAY

        float_array = decoder.decode_float_array()
        assert float_array == FLOAT_ARRAY
