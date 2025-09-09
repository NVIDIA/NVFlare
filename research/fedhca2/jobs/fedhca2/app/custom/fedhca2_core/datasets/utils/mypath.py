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

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]


class MyPath(object):
    """
    User-specific path configuration.
    """

    @staticmethod
    def db_root_dir(database=''):
        # Use absolute path to avoid issues with NVFLARE workspace copying
        db_root = '/home/suizhi/NVFlare/research/fedhca2/data'

        db_names = {'PASCALContext', 'NYUDv2'}

        if database in db_names:
            return os.path.join(db_root, database)

        elif not database:
            return db_root

        else:
            raise NotImplementedError
