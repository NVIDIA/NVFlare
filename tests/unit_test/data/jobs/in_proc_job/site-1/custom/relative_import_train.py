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

import argparse

from .model import Model


class Code:
    # code to test relative import
    def run(self, dataset_path, batch_size):
        model = Model()
        model.train(dataset_path, batch_size)

    def define_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_path", type=str, default="/data", nargs="?")
        parser.add_argument("--batch_size", type=int, default=4, nargs="?")
        return parser.parse_args()

    def main(self):
        args = self.define_parser()
        dataset_path = args.dataset_path
        batch_size = args.batch_size
        self.run(dataset_path, batch_size)


print("__name__ site-1 is ", __name__)

if __name__ == "__main__":
    code = Code()
    code.main()
