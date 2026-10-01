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

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext

REPO_ROOT = Path(__file__).resolve().parents[3]
TUTORIAL_ROOT = (
    Path("examples/tutorials/self-paced-training/part-5_federated_learning_applications_in_industries")
    / "chapter-11_federated_learning_in_healthcare_lifescience/11.2_drug_discovery"
)
AMPLIFY = Path("examples/advanced/amplify/src/filters.py")
TUTORIAL_AMPLIFY = TUTORIAL_ROOT / "11.2.2_drug_discovery_amplify/src/filters.py"
BIONEMO = Path("examples/advanced/bionemo/downstream/bionemo_filters.py")
TUTORIAL_BIONEMO = TUTORIAL_ROOT / "11.2.1_drug_discovery_bionemo/bionemo_filters.py"

EXCLUSION_FILTERS = [
    pytest.param((AMPLIFY, "ExcludeParamsFilter"), id="amplify"),
    pytest.param((TUTORIAL_AMPLIFY, "ExcludeParamsFilter"), id="tutorial-amplify"),
    pytest.param((BIONEMO, "BioNeMoExcludeParamsFilter"), id="bionemo-exclude"),
    pytest.param((TUTORIAL_BIONEMO, "BioNeMoExcludeParamsFilter"), id="tutorial-bionemo-exclude"),
]
DATA_FILTERS = EXCLUSION_FILTERS + [
    pytest.param((BIONEMO, "BioNeMoParamsFilter"), id="bionemo-prefix"),
    pytest.param((TUTORIAL_BIONEMO, "BioNeMoParamsFilter"), id="tutorial-bionemo-prefix"),
    pytest.param((BIONEMO, "BioNeMoStateDictFilter"), id="bionemo-state-dict"),
    pytest.param((TUTORIAL_BIONEMO, "BioNeMoStateDictFilter"), id="tutorial-bionemo-state-dict"),
    pytest.param(
        (Path("research/quantifying-data-leakage/src/nvflare_gradinv/filters/gaussian_privacy.py"), "GaussianPrivacy"),
        id="gaussian-privacy",
    ),
]


def _load_filter(path, class_name, **kwargs):
    spec = importlib.util.spec_from_file_location(f"bundled.filters.{class_name}", REPO_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)(**kwargs)


@pytest.fixture(params=DATA_FILTERS)
def data_filter(request):
    return _load_filter(*request.param)


@pytest.fixture(params=EXCLUSION_FILTERS)
def exclusion_filter(request):
    return _load_filter(*request.param)


@pytest.mark.parametrize("data_kind", [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF])
@pytest.mark.parametrize("in_collection", [False, True])
def test_data_filters_skip_empty_dxo(data_filter, data_kind, in_collection, monkeypatch):
    audit = Mock()
    monkeypatch.setattr("nvflare.apis.dxo_filter.add_job_audit_event", audit)
    dxo = DXO(data_kind=data_kind, data={}, meta={"keep": "metadata"})
    dxo.add_filter_history("previous-filter")
    root = DXO(data_kind=DataKind.COLLECTION, data={"model": dxo}) if in_collection else dxo
    shareable = root.to_shareable()

    result = data_filter.process(shareable, FLContext())
    result_dxo = from_shareable(result)
    if in_collection:
        result_dxo = result_dxo.data["model"]

    assert result is shareable
    assert result_dxo.data == {}
    assert result_dxo.meta == dxo.meta
    assert result_dxo.get_filter_history() == ["previous-filter"]
    audit.assert_not_called()


@pytest.mark.parametrize("data_kind", [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF])
def test_rdlv_filter_skips_empty_dxo_before_setup(data_kind, tmp_path, monkeypatch):
    # Dataset/inversion dependencies are not needed to check the empty-input path.
    dependencies = {
        name: MagicMock()
        for name in [
            "torch",
            "monai.data",
            "monai.transforms",
            "bundled.utils.rdlv_io",
            "bundled.filters.gradinv",
            "bundled.filters.image_sim",
        ]
    }
    with patch.dict(sys.modules, dependencies):
        filter_ = _load_filter(
            Path("research/quantifying-data-leakage/src/nvflare_gradinv/filters/rdlv_filter.py"),
            "RelativeDataLeakageValueFilter",
            data_root=str(tmp_path),
            dataset_json=__file__,
        )
    filter_._setup = Mock(side_effect=AssertionError("Empty input must not trigger dataset/inverter setup"))
    audit = Mock()
    monkeypatch.setattr("nvflare.apis.dxo_filter.add_job_audit_event", audit)
    shareable = DXO(data_kind=data_kind, data={}).to_shareable()

    result = filter_.process(shareable, FLContext())

    assert from_shareable(result).data == {}
    assert from_shareable(result).get_filter_history() is None
    filter_._setup.assert_not_called()
    audit.assert_not_called()


@pytest.mark.parametrize("data_kind", [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF])
def test_exclusion_filters_still_exclude_nonempty_data(exclusion_filter, data_kind, monkeypatch):
    monkeypatch.setattr("nvflare.apis.dxo_filter.add_job_audit_event", lambda **kwargs: None)
    excluded_key = f"{exclusion_filter.exclude_vars}.weight"
    shareable = DXO(data_kind=data_kind, data={"encoder.weight": 1, excluded_key: 2}).to_shareable()

    result = exclusion_filter.process(shareable, FLContext())

    assert from_shareable(result).data == {"encoder.weight": 1}
    assert from_shareable(result).get_filter_history() == [type(exclusion_filter).__name__]


@pytest.mark.parametrize("data_kind", [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF])
def test_exclusion_filters_still_reject_nonempty_data_without_matching_keys(exclusion_filter, data_kind):
    shareable = DXO(data_kind=data_kind, data={"encoder.weight": 1}).to_shareable()

    with pytest.raises(ValueError, match="did not match any exclude keys"):
        exclusion_filter.process(shareable, FLContext())


class _ModelInitializer(DXOFilter):
    def __init__(self, return_new_dxo):
        super().__init__(supported_data_kinds=[DataKind.WEIGHTS], data_kinds_to_filter=[DataKind.WEIGHTS])
        self.return_new_dxo = return_new_dxo

    def process_dxo(self, dxo, shareable, fl_ctx):
        if self.return_new_dxo:
            return DXO(data_kind=dxo.data_kind, data={"initialized.weight": 1}, meta=dxo.meta)
        dxo.data["initialized.weight"] = 1
        return dxo


@pytest.mark.parametrize("return_new_dxo", [False, True])
def test_skipping_empty_data_still_allows_initialization(exclusion_filter, return_new_dxo, monkeypatch):
    monkeypatch.setattr("nvflare.apis.dxo_filter.add_job_audit_event", lambda **kwargs: None)
    shareable = DXO(data_kind=DataKind.WEIGHTS, data={}).to_shareable()

    result = exclusion_filter.process(shareable, FLContext())
    result = _ModelInitializer(return_new_dxo).process(result, FLContext())

    assert from_shareable(result).data == {"initialized.weight": 1}
    assert from_shareable(result).get_filter_history() == ["_ModelInitializer"]
