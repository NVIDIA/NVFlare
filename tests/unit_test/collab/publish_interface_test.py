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

import copy

import pytest

from nvflare.collab.api.app import App
from nvflare.collab.api.decorators import publish
from nvflare.collab.api.publish_interface import PublishInterface


class _Target:
    @publish
    def train(self, model, /, *, rounds=1):
        return model


def test_publish_interface_is_immutable_and_round_trips_to_wire_dict():
    source = {"train": ["model", "context"], "stop": []}

    interface = PublishInterface.from_dict(source)
    source["train"].append("later")

    assert tuple(interface.get_method("train")) == ("model", "context")
    assert tuple(interface.get_method("stop")) == ()
    assert interface.get_method("missing") is None
    assert interface.to_dict() == {"train": ["model", "context"], "stop": []}
    assert copy.copy(interface) is interface
    assert copy.deepcopy(interface) is interface


def test_publish_interface_from_dict_preserves_existing_instance():
    interface = PublishInterface({"train": ["model"]})

    assert PublishInterface.from_dict(interface) is interface


def test_app_keeps_publish_interface_locally_and_exposes_wire_dict():
    app = App(_Target(), "target")

    assert isinstance(app.get_target_object_publish_interface("target"), PublishInterface)
    assert tuple(app.get_target_object_publish_interface("target").get_method("train")) == ("model", "rounds")
    assert app.get_collab_interface()["target"] == {
        "train": [
            {"name": "model", "kind": "POSITIONAL_ONLY", "required": True},
            {"name": "rounds", "kind": "KEYWORD_ONLY", "required": False},
        ]
    }


@pytest.mark.parametrize(
    "methods,error,match",
    [
        ([], TypeError, "methods must be a mapping"),
        ({1: []}, TypeError, "method name must be str"),
        ({"": []}, ValueError, "method name must not be empty"),
        ({"train": "model"}, TypeError, "must be a sequence"),
        ({"train": [1]}, TypeError, "parameter specification must be str or mapping"),
        ({"train": [""]}, ValueError, "parameter name.*must not be empty"),
        ({"train": ["model", "model"]}, ValueError, "duplicate parameter names"),
    ],
)
def test_publish_interface_rejects_invalid_schema(methods, error, match):
    with pytest.raises(error, match=match):
        PublishInterface(methods)
