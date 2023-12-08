# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from nvflare.fuel.utils.validation_utils import check_object_type


class ExportMode:
    SELF = "SELF"
    PEER = "PEER"


class AttributesExportable(ABC):
    """Export attributes."""

    @abstractmethod
    def export(self, export_mode: str) -> Tuple[str, dict]:
        """Exports attributes.

        Args:
            export_mode (str): export to peer (`ExportMode.PEER`) or to self (`ExportMode.SELF`).

        Returns:
            A tuple of (export section name, arguments to be exported)
        """
        pass


def export_components(
    components: Dict[str, AttributesExportable],
    reserved_keys: List[str],
    export_mode: str,
) -> dict:
    """Exports components.

    Args:
        components: A dict of {component_id: AttributesExportable}
        reserved_keys: keys that are reserved for system purpose.
        export_mode (str): export to peer (`ExportMode.PEER`) or to self (`ExportMode.SELF`).
    """
    components_data = {}
    for component_id, component_instance in components.items():
        check_object_type(component_id, component_instance, AttributesExportable)
        export_id, export_args = component_instance.export(export_mode)
        if export_id in components_data:
            raise RuntimeError(f"export_id {export_id} from {component_id} is duplicated, please change.")
        if export_id in reserved_keys:
            raise RuntimeError(f"export_id {export_id} from {component_id} is a reserved key, please change.")
        components_data[export_id] = export_args
    return components_data
