# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastapi import APIRouter

from .endpoints import get_apps, get_range_stats, get_stats, get_stats_list

api_router = APIRouter()
api_router.include_router(get_apps.router, prefix="/get_apps", tags=["get_apps"])
api_router.include_router(
    get_stats_list.router, prefix="/get_stats_list", tags=["get_stats_list"]
)
api_router.include_router(get_stats.router, prefix="/get_stats", tags=["get_stats"])
api_router.include_router(
    get_range_stats.router, prefix="/get_range_stats", tags=["get_range_stats"]
)
