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

"""Jinja2-based template engine for NVFlare provisioning."""

import os
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape


class TemplateEngine:
    """Jinja2-based template engine for NVFlare provisioning."""

    def __init__(self, template_dirs: Optional[List[str]] = None):
        if template_dirs is None:
            template_dirs = [os.path.join(os.path.dirname(__file__), "templates")]

        self.env = Environment(
            loader=FileSystemLoader(template_dirs),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def render(self, template_path: str, context: Dict[str, Any]) -> str:
        """Render a template file with the given context."""
        template = self.env.get_template(template_path)
        return template.render(**context)
