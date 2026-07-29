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

import os
import warnings
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.utils.constants import FrameworkType
from nvflare.recipe import PotentialSecretWarning, UnsupportedSecretRefWarning, secret_file_ref, secret_ref
from nvflare.recipe.fedavg import FedAvgRecipe

# Fake credential for testing the detector -- not a real token.
FAKE_GITHUB_TOKEN = "ghp_" + "Ab1" * 12


@pytest.fixture()
def make_recipe(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('training')\n")

    def _make_recipe(train_args="", **kwargs):
        return FedAvgRecipe(
            name="secrets-test-job",
            model=[1.0, 2.0],
            min_clients=2,
            num_rounds=1,
            train_script=str(script),
            train_args=train_args,
            framework=FrameworkType.NUMPY,
            **kwargs,
        )

    return _make_recipe


def _no_secret_warnings(record):
    return [w for w in record if issubclass(w.category, PotentialSecretWarning)] == []


class TestRecipeSecretScanning:
    def test_train_args_with_token_warns_without_leaking(self, make_recipe):
        recipe = make_recipe(train_args=f"--api_key {FAKE_GITHUB_TOKEN}")
        with pytest.warns(PotentialSecretWarning) as record:
            recipe._warn_potential_secrets_in_params()
        messages = [str(w.message) for w in record]
        assert any("train_args" in m for m in messages)
        assert all(FAKE_GITHUB_TOKEN not in m for m in messages)

    def test_external_command_with_token_warns_without_leaking(self, make_recipe):
        password = "hunter22x"
        recipe = make_recipe(
            launch_external_process=True,
            command=f"env API_PASSWORD={password} python3 -u",
        )
        with pytest.warns(PotentialSecretWarning) as record:
            recipe._warn_potential_secrets_in_params()
        messages = [str(w.message) for w in record]
        assert any("command" in message for message in messages)
        assert all(password not in message for message in messages)

    def test_external_command_secret_ref_assignment_is_safe(self, make_recipe):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            recipe = make_recipe(
                launch_external_process=True,
                command="env API_PASSWORD=${secret:API_PASSWORD} python3 -u",
            )
            recipe._warn_potential_secrets_in_params()
        assert _no_secret_warnings(record)

    def test_clean_train_args_no_warning(self, make_recipe):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            recipe = make_recipe(train_args="--epochs 5 --lr 0.1 --data_path /data/site-1")
            recipe._warn_potential_secrets_in_params()
        assert _no_secret_warnings(record)

    def test_per_site_config_with_password_warns(self, make_recipe):
        recipe = make_recipe()
        with pytest.warns(PotentialSecretWarning):
            recipe.set_per_site_config(
                {
                    "site-1": {"train_args": "--password hunter22x"},
                    "site-2": {"train_args": "--epochs 5"},
                }
            )
        with pytest.warns(PotentialSecretWarning) as record:
            recipe._warn_potential_secrets_in_params()
        messages = [str(w.message) for w in record]
        assert any("per_site_config" in m for m in messages)
        assert all("hunter22x" not in m for m in messages)

    def test_set_per_site_config_warns(self, make_recipe):
        recipe = make_recipe()
        with pytest.warns(PotentialSecretWarning):
            recipe.set_per_site_config({"site-1": {"api_key": "abcd1234efgh"}, "site-2": {}})

    def test_add_client_config_warns(self, make_recipe):
        recipe = make_recipe()
        with pytest.warns(PotentialSecretWarning):
            recipe.add_client_config({"auth_token": "abcd1234efgh"})

    def test_add_server_config_warns(self, make_recipe):
        recipe = make_recipe()
        with pytest.warns(PotentialSecretWarning):
            recipe.add_server_config({"password": "abcd1234efgh"})

    @pytest.mark.parametrize("method_name", ["add_client_config", "add_server_config"])
    def test_add_config_warns_for_nested_secret_ref_key(self, make_recipe, method_name):
        recipe = make_recipe()
        config = {"service": {secret_ref("DYNAMIC_KEY"): "value"}}

        with pytest.warns(UnsupportedSecretRefWarning, match=method_name):
            getattr(recipe, method_name)(config)

    def test_exec_params_warn_on_export(self, make_recipe, tmp_path):
        recipe = make_recipe()
        with pytest.warns(PotentialSecretWarning) as record:
            recipe.export(str(tmp_path), client_exec_params={"api_key": "abcd1234efgh"})
        assert any("client_exec_params" in str(w.message) for w in record)

    @pytest.mark.parametrize("params_name", ["server_exec_params", "client_exec_params"])
    def test_exec_params_warn_for_nested_secret_ref_key(self, make_recipe, tmp_path, params_name):
        recipe = make_recipe()
        params = {"service": {secret_ref("DYNAMIC_KEY"): "value"}}

        with pytest.warns(UnsupportedSecretRefWarning, match=params_name):
            recipe.export(str(tmp_path), **{params_name: params})

    def test_export_scans_generated_config_files(self, make_recipe, tmp_path):
        recipe = make_recipe(train_args=f"--api_key {FAKE_GITHUB_TOKEN}")
        with pytest.warns(PotentialSecretWarning) as record:
            recipe.export(str(tmp_path))
        messages = [str(w.message) for w in record]
        assert any("exported job file" in m for m in messages)
        assert all(FAKE_GITHUB_TOKEN not in m for m in messages)

    def test_run_rescans_recipe_parameters(self, make_recipe):
        recipe = make_recipe()
        recipe.train_args = f"--api-key {FAKE_GITHUB_TOKEN}"
        env = MagicMock()
        env.deploy.return_value = "job-id"

        with pytest.warns(PotentialSecretWarning, match="train_args"):
            run = recipe.run(env)

        assert run.get_job_id() == "job-id"

    def test_warning_ignored_during_construction_is_emitted_on_run(self, make_recipe):
        env = MagicMock()
        env.deploy.return_value = "job-id"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PotentialSecretWarning)
            recipe = make_recipe(train_args=f"--api-key {FAKE_GITHUB_TOKEN}")

        with pytest.warns(PotentialSecretWarning, match="train_args"):
            recipe.run(env)


class TestSecretRefEndToEnd:
    def test_secret_ref_in_train_args_no_warning_and_placeholder_in_export(self, make_recipe, tmp_path):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            recipe = make_recipe(train_args=f"--epochs 5 --api-key {secret_ref('MY_API_KEY')}")
            recipe.export(str(tmp_path))
        assert _no_secret_warnings(record)

        client_configs = []
        for root, _dirs, files in os.walk(tmp_path):
            for file_name in files:
                if file_name == "config_fed_client.json":
                    with open(os.path.join(root, file_name)) as f:
                        client_configs.append(f.read())
        assert client_configs
        assert any("${secret:MY_API_KEY}" in config for config in client_configs)

    def test_secret_file_ref_is_public_and_placeholder_is_exported(self, make_recipe, tmp_path):
        placeholder = secret_file_ref("/var/run/secrets/my-app/api-key")
        recipe = make_recipe(train_args=f"--api-key {placeholder}")

        recipe.export(str(tmp_path))

        client_configs = []
        for root, _dirs, files in os.walk(tmp_path):
            for file_name in files:
                if file_name == "config_fed_client.json":
                    with open(os.path.join(root, file_name)) as f:
                        client_configs.append(f.read())
        assert client_configs
        assert any(placeholder in config for config in client_configs)
