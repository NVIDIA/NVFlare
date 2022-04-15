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

import json
import re

from nvflare.fuel.utils.json_scanner import JsonObjectProcessor, JsonScanner, Node

test_json = """
{
 "learning_rate": 1e-4,
 "lr_search" : [1e-4, 2e-3],
 "train": {
  "model": {
    "name": "SegAhnet",
    "args": {
      "num_classes": 2,
      "if_use_psp": false,
      "pretrain_weight_name": "{PRETRAIN_WEIGHTS_FILE}",
      "plane": "z",
      "final_activation": "softmax",
      "n_spatial_dim": 3
    },
    "search": [
      {
        "type": "float",
        "args": ["num_classes"],
        "targets": [1,3],
        "domain": "net"
      },
      {
        "type": "float",
        "args": ["n_spatial_dim"],
        "targets": [2,5],
        "domain": "net"
      },
      {
        "type": "enum",
        "args": ["n_spatial_dim", "num_classes"],
        "targets": [[2,3],[3,4],[5,1]],
        "domain": "net"
      },
      {
        "type": "enum",
        "args": ["n_spatial_dim"],
        "targets": [[2],[3],[6],[12]],
        "domain": "net"
      }
    ]
  },
  "pre_transforms": [
  {
    "name": "LoadNifti",
    "args": {
      "fields": [
        "image",
        "label"
      ]
    }
  },
  {
    "name": "ConvertToChannelsFirst",
    "args": {
      "fields": [
        "image",
        "label"
      ]
    }
  },
  {
    "name": "ScaleIntensityRange",
    "args": {
      "fields": "image",
      "a_min": -57,
      "a_max": 164,
      "b_min": 0.0,
      "b_max": 1.0,
      "clip": true
    }
  },
  {
    "name": "FastCropByPosNegRatio",
    "args": {
      "size": [
        96,
        96,
        96
      ],
      "fields": "image",
      "label_field": "label",
      "pos": 1,
      "neg": 1,
      "batch_size": 3
    },
    "search": [
      {
        "domain": "transform",
        "type": "enum",
        "args": ["size"],
        "targets": [[[32, 32, 32]], [[64, 64, 64]], [[128, 128, 128]]]
      },
      {
        "domain": "transform",
        "type": "enum",
        "args": ["batch_size"],
        "targets": [[3], [4], [8], [10]]
      }
    ]
  },
  {
    "name": "RandomAxisFlip",
    "args": {
      "fields": [
        "image",
        "label"
      ],
      "probability": 0.0
    },
    "search": [
      {
        "domain": "transform",
        "type": "float",
        "args": ["probability#p"],
        "targets": [0.0, 1.0]
      },
      {
        "domain": "transform",
        "args": "DISABLED"
      }
    ]
  },
  {
    "name": "RandomRotate3D",
    "args": {
      "fields": [
        "image",
        "label"
      ],
      "probability": 0.0
    }
  },
  {
    "name": "ScaleIntensityOscillation",
    "args": {
      "fields": "image",
      "magnitude": 0.10
    }
  },
  {
    "name": "LoadNifti",
    "args": {
      "fields": [
        "image",
        "label"
      ]
    }
  },
  {
    "name": "LoadNifti",
    "args": {
      "fields": [
        "image",
        "label"
      ]
    }
  },
  {
    "name": "LoadNifti",
    "args": {
      "fields": [
        "image",
        "label"
      ]
    }
  },
  {
    "name": "LoadNifti",
    "args": {
      "fields": [
        "image",
        "label"
      ]
    }
  },
  {
    "name": "RandomAxisFlip",
    "args": {
      "fields": [
        "image",
        "label"
      ],
      "probability": 0.0
    },
    "search": [
      {
        "domain": "transform",
        "type": "float",
        "args": ["probability#p"],
        "targets": [0.0, 1.0]
      },
      {
        "domain": "transform",
        "args": "DISABLED"
      }
    ]
  }
]
} }
"""
TRAIN_CONFIG = json.loads(test_json)


def _post_process_element(node: Node):
    path = node.path()
    print("EXIT Level: {}; Key: {}; Pos: {}; Path: {}".format(node.level, node.key, node.position, path))


class _TestJsonProcessor(JsonObjectProcessor):
    def process_element(self, node: Node):
        pats = [
            r".\.pre_transforms\.#[0-9]+$",
            r"^train\.model\.name$",
            r".\.search\.#[0-9]+$",
            r".\.pre_transforms\.#[0-9]+\.args$",
        ]
        path = node.path()
        print("ENTER Level: {}; Key: {}; Pos: {}; Path: {}".format(node.level, node.key, node.position, path))
        for p in pats:
            x = re.search(p, path)
            if x:
                print("\t {} matches {}".format(path, p))

        node.exit_cb = _post_process_element


class TestJsonScanner:
    def test_scan(self):
        scanner = JsonScanner(TRAIN_CONFIG, "test")
        scanner.scan(_TestJsonProcessor())
