# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


class FileSource:
    def __init__(self, src_path: str, dest_dir=None, app_folder_type=None):
        """Constructor of FileSource
        A FileSource defines additional file(s) to be added to the job app's app_folder_type folder
        when generating the job config.

        Args:
            src_path: the path to the source, which could be a single file or a directory.
            dest_dir: the relative dir within the app_folder folder that the copied source will be placed.
            app_folder_type: the type of the app folder to place the copied files.
                Valid values are: custom, config
                If not specified, default to "custom"

        Note: a FileSource could be any type of files, not limited to "py" files!
        Even if the source is a python script, no special processing (e.g. package import scanning) will be done.
        """
        self.src_path = src_path
        self.dest_dir = dest_dir
        self.app_folder_type = app_folder_type

        if app_folder_type:
            valid_folder_types = ["custom", "config"]
            if app_folder_type not in valid_folder_types:
                raise ValueError(
                    f"invalid value of 'app_folder_type' ({app_folder_type}): must be one of {valid_folder_types}"
                )

        if not os.path.exists(src_path):
            raise ValueError(f"src_path {src_path} does not exist")

        if dest_dir:
            if not isinstance(dest_dir, str):
                raise ValueError(f"dest_dir must be str but got {type(dest_dir)}")

            if os.path.isabs(dest_dir):
                raise ValueError(f"dest_dir {dest_dir} must not be absolute")

    def add_to_fed_job(self, job, ctx, **kwargs):
        """This method is used by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:

        """
        job.check_kwargs(args_to_check=kwargs, args_expected={})
        job.add_file_source(
            src_path=self.src_path,
            dest_dir=self.dest_dir,
            ctx=ctx,
            app_folder_type=self.app_folder_type,
        )
