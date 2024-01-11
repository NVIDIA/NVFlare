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

import os

from nvflare.lighter.spec import Builder
from nvflare.lighter.utils import load_yaml


class TemplateBuilder(Builder):
    """Load template file.

    Loads the content of the template_file into the key-value pair (template) in the build context.
    """

    def initialize(self, ctx):
        resource_dir = self.get_resources_dir(ctx)
        template_files = ctx.get("template_files")
        template = dict()
        for tplt_file in template_files:
            template.update(load_yaml(os.path.join(resource_dir, tplt_file)))
        ctx["template"] = template
