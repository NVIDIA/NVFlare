# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import pathlib
import shutil
import subprocess

from nvflare.lighter.spec import Builder, Project
from nvflare.lighter.utils import generate_password


class WorkspaceBuilder(Builder):
    def __init__(self, template_file):
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
            template_file: name of template file containing scripts and configs to put into startup folders
        """
        self.template_file = template_file

    def _make_dir(self, dirs):
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def initialize(self, ctx):
        workspace_dir = ctx["workspace"]
        prod_dirs = [_ for _ in os.listdir(workspace_dir) if _.startswith("prod_")]
        last = -1
        for dir in prod_dirs:
            stage = int(dir.split("_")[-1])
            if stage > last:
                last = stage
        ctx["last_prod_stage"] = last
        template_file_full_path = os.path.join(self.get_resources_dir(ctx), self.template_file)
        file_path = pathlib.Path(__file__).parent.absolute()
        shutil.copyfile(os.path.join(file_path, self.template_file), template_file_full_path)
        ctx["template_file"] = self.template_file

    def build(self, project: Project, ctx: dict):
        dirs = [self.get_kit_dir(p, ctx) for p in project.participants]
        self._make_dir(dirs)

    def finalize(self, ctx: dict):
        if ctx["last_prod_stage"] >= 99:
            print(f"Please clean up {ctx['workspace']} by removing prod_N folders")
            print("After clean-up, rerun the provision command.")
        else:
            current_prod_stage = str(ctx["last_prod_stage"] + 1).zfill(2)
            current_prod_dir = os.path.join(ctx["workspace"], f"prod_{current_prod_stage}")
            shutil.move(self.get_wip_dir(ctx), current_prod_dir)
            ctx.pop("wip_dir", None)
            print(f"Generated results can be found under {current_prod_dir}.  Builder's wip folder removed.")
            ctx["current_prod_dir"] = current_prod_dir


class DistributionBuilder(Builder):
    def __init__(self, zip_password=False):
        """Build the zip files for each folder.

        Creates the zip files containing the archives for each startup kit. It will add password protection if the
        argument (zip_password) is true.

        Args:
            zip_password: if true, will create zipped packages with passwords
        """
        self.zip_password = zip_password

    def build(self, project: Project, ctx: dict):
        """Create a zip for each individual folder.
        Note that if zip_password is True, the zip command will be used to encrypt zip files.  Users have to to
        install this zip utility before provisioning.  In Ubuntu system, use this command to install zip utility:
        sudo apt-get install zip

        Args:
            project (Project): project instance
            ctx (dict): the provision context
        """
        wip_dir = self.get_wip_dir(ctx)
        dirs = [name for name in os.listdir(wip_dir) if os.path.isdir(os.path.join(wip_dir, name))]
        for dir in dirs:
            dest_zip_file = os.path.join(wip_dir, f"{dir}")
            if self.zip_password:
                pw = generate_password()
                run_args = ["zip", "-rq", "-P", pw, dest_zip_file + ".zip", ".", "-i", "startup/*"]
                os.chdir(dest_zip_file)
                try:
                    subprocess.run(run_args)
                    print(f"Password {pw} on {dir}.zip")
                except FileNotFoundError as e:
                    raise RuntimeError("Unable to zip folders with password.  Maybe the zip utility is not installed.")
                finally:
                    os.chdir(os.path.join(dest_zip_file, ".."))
            else:
                shutil.make_archive(dest_zip_file, "zip", root_dir=os.path.join(wip_dir, dir), base_dir="startup")
