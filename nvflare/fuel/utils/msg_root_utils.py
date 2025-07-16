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
from nvflare.apis.fl_constant import ReservedTopic
from nvflare.fuel.data_event.data_bus import DataBus


def _make_topic(base_topic: str, msg_root_id: str) -> str:
    return f"{base_topic}_{msg_root_id}"


def delete_msg_root(msg_root_id: str):
    topic = _make_topic(ReservedTopic.MSG_ROOT_DELETED, msg_root_id)
    DataBus().publish([topic], datum=msg_root_id)


def subscribe_to_msg_root(msg_root_id: str, cb, **cb_kwargs):
    topic = _make_topic(ReservedTopic.MSG_ROOT_DELETED, msg_root_id)
    DataBus().subscribe(topics=[topic], callback=_msg_root_deleted, app_cb=cb, **cb_kwargs)


def _msg_root_deleted(topic: str, msg_root_id: str, db: DataBus, app_cb, **cb_kwargs):
    app_cb(msg_root_id, **cb_kwargs)
    db.unsubscribe(topic)
