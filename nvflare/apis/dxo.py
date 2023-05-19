# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Union

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.utils import fobs


class DataKind(object):
    WEIGHTS = "WEIGHTS"
    WEIGHT_DIFF = "WEIGHT_DIFF"
    XGB_MODEL = "XGB_MODEL"
    METRICS = "METRICS"
    ANALYTIC = "ANALYTIC"
    COLLECTION = "COLLECTION"  # Dict or List of DXO objects
    STATISTICS = "STATISTICS"
    PSI = "PSI"


class MetaKey(object):
    NUM_STEPS_CURRENT_ROUND = "NUM_STEPS_CURRENT_ROUND"
    PROCESSED_ALGORITHM = "PROCESSED_ALGORITHM"
    PROCESSED_KEYS = "PROCESSED_KEYS"
    INITIAL_METRICS = "initial_metrics"
    FILTER_HISTORY = "filter_history"


_KEY_KIND = "kind"
_KEY_DATA = "data"
_KEY_META = "meta"
_KEY_DXO = "DXO"


class DXO(object):
    def __init__(self, data_kind: str, data: dict, meta: dict = None):
        """Init the DXO.

        The Data Exchange Object standardizes the data passed between communicating parties.

        Args:
            data_kind: kind of data
            data: clear-text data
            meta: None or dict for any additional properties
        """
        if data is None:
            data = {}
        if meta is None:
            meta = {}

        self.data_kind = data_kind
        self.data = data
        self.meta = meta

        err = self.validate()
        if err:
            raise ValueError("invalid DXO: {}".format(err))

    def get_meta_prop(self, key: str, default=None):
        if self.meta and isinstance(self.meta, dict):
            return self.meta.get(key, default)
        return default

    def set_meta_prop(self, key: str, value):
        if self.meta is None:
            self.meta = {}
        self.meta[key] = value

    def remove_meta_props(self, keys: List[str]):
        if self.meta and keys:
            for k in keys:
                self.meta.pop(k, None)

    def get_meta_props(self):
        return self.meta

    def update_meta_props(self, meta):
        self.meta.update(copy.deepcopy(meta))

    def _encode(self) -> dict:
        return {_KEY_KIND: self.data_kind, _KEY_DATA: self.data, _KEY_META: self.meta}

    def update_shareable(self, s: Shareable) -> Shareable:
        s.set_header(key=ReservedHeaderKey.CONTENT_TYPE, value="DXO")
        s[_KEY_DXO] = self._encode()
        return s

    def to_shareable(self) -> Shareable:
        """Convert the DXO object into Shareable.

        Returns:
            Shareable object.

        """
        s = Shareable()
        return self.update_shareable(s)

    def to_bytes(self) -> bytes:
        """Serialize the DXO object into bytes.

        Returns:
            object serialized in bytes.

        """
        return fobs.dumps(self)

    def validate(self) -> str:
        if self.data is None:
            return "missing data"

        if not isinstance(self.data, dict):
            return "invalid data: expect dict but got {}".format(type(self.data))

        if self.meta is not None and not isinstance(self.meta, dict):
            return "invalid props: expect dict but got {}".format(type(self.meta))

        return ""

    def add_filter_history(self, filter_name: Union[str, List[str]]):
        if not filter_name:
            return
        hist = self.get_meta_prop(MetaKey.FILTER_HISTORY)
        if not hist:
            hist = []
            self.set_meta_prop(MetaKey.FILTER_HISTORY, hist)
        if isinstance(filter_name, str):
            hist.append(filter_name)
        elif isinstance(filter_name, list):
            hist.extend(filter_name)

    def get_filter_history(self):
        return self.get_meta_prop(MetaKey.FILTER_HISTORY)


def from_shareable(s: Shareable) -> DXO:
    """Convert Shareable into a DXO object.

    Args:
        s: Shareable object

    Returns:
        DXO object.

    """
    content_type = s.get_header(ReservedHeaderKey.CONTENT_TYPE)
    if not content_type or content_type != "DXO":
        raise ValueError("the shareable is not a valid DXO - expect content_type DXO but got {}".format(content_type))

    encoded = s.get(_KEY_DXO, None)
    if not encoded:
        raise ValueError("the shareable is not a valid DXO - missing content")

    if not isinstance(encoded, dict):
        raise ValueError(
            "the shareable is not a valid DXO - should be encoded as dict but got {}".format(type(encoded))
        )

    k = encoded.get(_KEY_KIND, None)
    d = encoded.get(_KEY_DATA, None)
    m = encoded.get(_KEY_META, None)

    return DXO(data_kind=k, data=d, meta=m)


def from_bytes(data: bytes) -> DXO:
    """Convert the data bytes into Model object.

    Args:
        data: a bytes object

    Returns:
        an object loaded by FOBS from data

    """
    x = fobs.loads(data)
    if isinstance(x, DXO):
        return x
    else:
        raise ValueError("Data bytes are from type {} and do not represent a valid DXO instance.".format(type(x)))
