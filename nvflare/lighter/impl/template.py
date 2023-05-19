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

import json
import os

from nvflare.lighter.spec import Builder
from nvflare.lighter.utils import load_yaml


class TemplateBuilder(Builder):
    """Load template file.

    Loads the content of the template_file and the authz_def (section of template file with fixed authorization
    definitions) into two key-value pairs in the build context.
    """

    def initialize(self, ctx):
        resource_dir = self.get_resources_dir(ctx)
        template_file = ctx.get("template_file")
        template = load_yaml(os.path.join(resource_dir, template_file))
        authz_def = json.loads(template.get("authz_def"))
        ctx["template"] = template
        ctx["authz_def"] = authz_def
