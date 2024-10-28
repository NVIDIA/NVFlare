# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Flask configuration variables."""
import os
from datetime import timedelta

from nvflare.lighter.utils import generate_password


class Config:
    # General Config
    SECRET_KEY = os.environ.get("SECRET_KEY", generate_password(16))
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=30)

    # Database
    web_root = os.environ.get("NVFL_WEB_ROOT", "/var/tmp/nvflare/dashboard")
    default_sqlite_file = os.path.join(web_root, "db.sqlite")
    default_sqlite_url = f"sqlite:///{default_sqlite_file}"
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", default_sqlite_url)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
