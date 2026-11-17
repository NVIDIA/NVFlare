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

"""Site side of the catalog: one ODCS contract per dataset the site declares for the job's Study.

The site's ``local/catalog.yaml`` names each dataset and how its contract is produced::

    studies:
      oncology:
        datasets:
          omop:                      # provided: a file the data steward maintains
            contract: contracts/omop.odcs.yaml
          claims:                    # generated: read from the live source by the datacontract library
            import: {format: postgres, source: db.example.org, database: claims, schema: public}
            env: {DATACONTRACT_POSTGRES_USERNAME: CLAIMS_DB_USER, DATACONTRACT_POSTGRES_PASSWORD: CLAIMS_DB_PASSWORD}

``import`` holds the arguments of ``DataContract.import_from_source``. ``env`` maps the library's credential
settings to the site's environment variables that hold them.
"""

import logging
import os
from pathlib import Path

import yaml

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import get_job_meta_study

CONFIG_FILE = "catalog.yaml"
GENERATED_CONTRACT_KEYS = ("apiVersion", "kind", "id", "version", "status", "name", "schema")

_logger = logging.getLogger(__name__)


def read_site_catalog(fl_ctx: FLContext, config_root: str | None = None) -> dict:
    """The contracts of the running job's site and Study. ``config_root`` holds one folder per simulated site."""
    site = fl_ctx.get_identity_name()
    if config_root is None:
        config_dir = fl_ctx.get_workspace().get_site_config_dir()
    elif fl_ctx.get_prop(FLContextKey.SIMULATE_MODE, False):
        config_dir = os.path.join(config_root, site)
    else:
        raise RuntimeError("config_root is for the simulator only; a deployed site owns its configuration")
    return read_contracts(config_dir, site, get_job_meta_study(fl_ctx.get_prop(FLContextKey.JOB_META)))


def read_contracts(config_dir: str, site: str, study: str) -> dict:
    """``{"contracts": {dataset: ODCS YAML text}, "unavailable": [dataset]}`` for the datasets declared for ``study``."""
    contracts, unavailable = {}, []
    for name, spec in _declared_datasets(config_dir, study).items():
        name = str(name)
        try:
            if "contract" in spec:
                contracts[name] = Path(config_dir, spec["contract"]).read_text(encoding="utf-8")
            else:
                contracts[name] = _generate(site, name, spec["import"], spec.get("env", {}))
        except Exception as e:
            _logger.warning("catalog: dataset %r is unavailable (%s)", name, type(e).__name__)
            unavailable.append(name)
    return {"contracts": contracts, "unavailable": unavailable}


def _declared_datasets(config_dir: str, study: str) -> dict:
    path = os.path.join(config_dir, CONFIG_FILE)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["studies"].get(study, {}).get("datasets", {})


def _generate(site: str, name: str, import_args: dict, env: dict) -> str:
    from datacontract.data_contract import DataContract

    credentials = {setting: os.environ[variable] for setting, variable in env.items()}
    odcs = DataContract.import_from_source(config=credentials or None, **import_args)
    contract = yaml.safe_load(odcs.to_yaml())
    contract.update(id=f"{site}.{name}", name=name)
    schema_only = {key: contract[key] for key in GENERATED_CONTRACT_KEYS if key in contract}
    return yaml.safe_dump(schema_only, sort_keys=False)
