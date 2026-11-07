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

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import yaml
from sentence_transformers import SentenceTransformer


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def load_json(file_name: str) -> Union[list, dict]:

    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False, separators=(",", ": "))


def random_divide_list(lst: list[Any], k: int) -> list[list]:
    """
    Divides the list into chunks, each with maximum length k.

    Args:
        lst: The list to be divided.
        k: The maximum length of each chunk.

    Returns:
        A list of chunks.
    """
    if len(lst) == 0:
        return []

    random.shuffle(lst)
    if len(lst) <= k:
        return [lst]
    else:
        num_chunks = math.ceil(len(lst) / k)
        chunk_size = math.ceil(len(lst) / num_chunks)
        return [lst[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]


_EMBEDDING_MODEL_CACHE = {}


@dataclass
class EmbeddingFunc:

    model_type: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self):
        if self.model_type not in _EMBEDDING_MODEL_CACHE:
            _EMBEDDING_MODEL_CACHE[self.model_type] = SentenceTransformer(self.model_type)

        self.func: SentenceTransformer = _EMBEDDING_MODEL_CACHE[self.model_type]

    def embed_documents(self, texts: list[str]) -> list[list]:
        return [self.func.encode(text).tolist() for text in texts]

    def embed_query(self, query: str) -> list:
        return self.func.encode(query).tolist()

    def embed_text(self, text: str) -> list:
        return self.func.encode(text).tolist()


@dataclass
class OTEmbeddingFunc(EmbeddingFunc):
    ot_ref_morph_path: str = ""
    ot_ref_normal_path: str = ""
    ot_map_path: str = ""
    ot_force_recompute: bool = False
    ot_ref_field: str = "claim"

    def __post_init__(self):
        super().__post_init__()
        self._ot_map = self._load_or_build_ot_map()

    def embed_query(self, query: str) -> list:
        import numpy as np

        embedding = np.asarray(self.func.encode(query), dtype=np.float32)
        aligned = embedding @ self._ot_map
        return aligned.astype(np.float32).tolist()

    def embed_text(self, text: str) -> list:
        return self.func.encode(text).tolist()

    def _load_or_build_ot_map(self) -> np.ndarray:
        import numpy as np

        try:
            import ot
        except Exception as exc:
            raise ModuleNotFoundError("OT mapping requires POT. Install with `pip install pot`.") from exc
        if self.ot_map_path:
            map_path = Path(self.ot_map_path)
            if map_path.exists() and not self.ot_force_recompute:
                data = np.load(map_path, allow_pickle=False)
                return data["ot_map"]
        else:
            map_path = None

        morph_claims = self._load_jsonl_claims(Path(self.ot_ref_morph_path), self.ot_ref_field)
        normal_claims = self._load_jsonl_claims(Path(self.ot_ref_normal_path), self.ot_ref_field)
        if not morph_claims or not normal_claims:
            raise ValueError("OT reference datasets are empty or missing.")

        morph_embeddings = np.asarray(self.embed_documents(morph_claims), dtype=np.float32)
        normal_embeddings = np.asarray(self.embed_documents(normal_claims), dtype=np.float32)

        n_morph = morph_embeddings.shape[0]
        n_normal = normal_embeddings.shape[0]
        a = np.full(n_morph, 1.0 / n_morph, dtype=np.float64)
        b = np.full(n_normal, 1.0 / n_normal, dtype=np.float64)
        cost = ot.dist(morph_embeddings, normal_embeddings)
        gamma = ot.emd(a, b, cost)
        aligned_morph = n_morph * gamma @ normal_embeddings

        ot_map, _, _, _ = np.linalg.lstsq(morph_embeddings, aligned_morph, rcond=None)
        if map_path is not None:
            map_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(map_path, ot_map=ot_map, n_morph=n_morph, n_normal=n_normal)
        return ot_map.astype(np.float32)

    @staticmethod
    def _load_jsonl_claims(path: Path, field: str = "claim") -> list[str]:
        claims: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                claim = obj.get(field)
                if claim is None:
                    continue
                if isinstance(claim, dict):
                    claim = json.dumps(claim, ensure_ascii=False, separators=(",", ":"))
                claims.append(str(claim))
        return claims
