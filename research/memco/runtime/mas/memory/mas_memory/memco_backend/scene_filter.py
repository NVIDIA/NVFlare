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

import os
from typing import Iterable, List, Optional, Sequence, Set

SCENE_TYPE_RANGES = {
    "kitchen": range(1, 31),
    "living_room": range(201, 231),
    "bedroom": range(301, 331),
    "bathroom": range(401, 431),
}


def parse_csv_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_scene_ids(value: Optional[str]) -> Set[int]:
    ids = set()
    for item in parse_csv_arg(value):
        ids.add(int(item))
    return ids


def normalize_scene_types(scene_types: Sequence[str]) -> List[str]:
    normalized = []
    for scene_type in scene_types:
        key = scene_type.strip().lower()
        if key not in SCENE_TYPE_RANGES:
            raise ValueError(
                f"Unknown scene type '{scene_type}'. Expected one of: " f"{', '.join(SCENE_TYPE_RANGES.keys())}"
            )
        normalized.append(key)
    return normalized


def extract_scene_num_from_game_path(game_path: str) -> Optional[int]:
    parts = os.path.normpath(game_path).split(os.sep)
    if len(parts) < 3:
        return None
    task_dir = parts[-3]
    try:
        return int(task_dir.rsplit("-", 1)[-1])
    except ValueError:
        return None


def scene_type_from_scene_num(scene_num: Optional[int]) -> Optional[str]:
    if scene_num is None:
        return None
    for scene_type, scene_range in SCENE_TYPE_RANGES.items():
        if scene_num in scene_range:
            return scene_type
    return None


def game_matches_scene_filters(
    game_path: str,
    allowed_scene_types: Optional[Iterable[str]] = None,
    allowed_scene_ids: Optional[Iterable[int]] = None,
) -> bool:
    scene_num = extract_scene_num_from_game_path(game_path)
    scene_type = scene_type_from_scene_num(scene_num)

    if allowed_scene_types:
        if scene_type not in set(allowed_scene_types):
            return False
    if allowed_scene_ids:
        if scene_num not in set(allowed_scene_ids):
            return False
    return True


def filter_game_list_by_scene(
    game_list,
    scene_types_arg: Optional[str] = None,
    scene_ids_arg: Optional[str] = None,
):
    scene_types = normalize_scene_types(parse_csv_arg(scene_types_arg))
    scene_ids = parse_scene_ids(scene_ids_arg)

    if not scene_types and not scene_ids:
        return list(game_list)

    return [
        game_path
        for game_path in game_list
        if game_matches_scene_filters(
            game_path,
            allowed_scene_types=scene_types or None,
            allowed_scene_ids=scene_ids or None,
        )
    ]
