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
import shutil

from nvflare.lighter.constants import CtxKey
from nvflare.lighter.spec import Builder, Project, ProvisionContext
from nvflare.lighter.utils import make_dirs


class WorkspaceBuilder(Builder):
    def __init__(self, template_file=None):
        """Manages the folder structure for provisioned projects.

        Sets the template_file containing scripts and configs to put into startup folders, creates directories for the
        participants, and moves the provisioned project to the final location at the end
        ($WORKSPACE/$PROJECT_NAME/prod_XX). WorkspaceBuilder manages and sets the number in prod_XX by incrementing from
        the last time provision was run for this project in this workspace, starting with 00 to a max of 99.

        Each time the provisioning tool runs, it requires a workspace folder in the local file system.  The workspace
        will have the following folder structure:

        .. code-block:: text

          $WORKSPACE/    <--- this is assigned by -w option of provision command (default is workspace)
            $PROJECT_NAME/  <--- this is the name value in the project.yml file
              prod_00/   <--- a new prod_NN folder is created if provision does not have any errors.
              prod_01/
              ...
              resources/ <--- this folder stores resources for other builders to load
              state/     <--- this folder stores persistent information (such as certificates) so subsequent runs of the provision command can load the state back.
              wip/  <--- this is only used during runtime, and will be removed when the provision command exits

        Args:
            template_file: one or more template file names
        """
        self.template_files = template_file  # obsolete

    def initialize(self, project: Project, ctx: ProvisionContext):
        # be backward compatible: load template files if specified.
        ctx.load_templates(self.template_files)

        workspace_dir = ctx.get_workspace()
        prod_dirs = [_ for _ in os.listdir(workspace_dir) if _.startswith("prod_")]
        last = -1
        for d in prod_dirs:
            stage = int(d.split("_")[-1])
            if stage > last:
                last = stage
        ctx[CtxKey.LAST_PROD_STAGE] = last

    def build(self, project: Project, ctx: ProvisionContext):
        participants = project.get_all_participants()
        dirs = [ctx.get_kit_dir(p) for p in participants]
        make_dirs(dirs)

        dirs = [ctx.get_transfer_dir(p) for p in participants]
        make_dirs(dirs)

        dirs = [ctx.get_local_dir(p) for p in participants]
        make_dirs(dirs)

    def finalize(self, project: Project, ctx: ProvisionContext):
        if ctx[CtxKey.LAST_PROD_STAGE] >= 99:
            ctx.info(f"Please clean up {ctx['workspace']} by removing prod_N folders")
            ctx.info("After clean-up, rerun the provision command.")
        else:
            current_prod_stage = str(ctx[CtxKey.LAST_PROD_STAGE] + 1).zfill(2)
            current_prod_dir = os.path.join(ctx.get_workspace(), f"prod_{current_prod_stage}")
            shutil.move(ctx.get_wip_dir(), current_prod_dir)
            ctx.pop(CtxKey.WIP, None)
            ctx.info(f"Generated results can be found under {current_prod_dir}. ")
            ctx[CtxKey.CURRENT_PROD_DIR] = current_prod_dir
