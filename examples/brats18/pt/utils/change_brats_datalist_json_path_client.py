# Copyright (c) 2021, NVIDIA CORPORATION.
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
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="generate train_config for brats")
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--run_number", type=int)
    parser.add_argument("--num_sites", type=int)
    parser.add_argument("--site_pre", type=str)
    args = parser.parse_args()

    for client_id in range(1, args.num_sites+1):           
        out_env_dir = os.path.join(
            args.base_dir,
            args.site_pre+str(client_id),
            "run_"+str(args.run_number),
            "app_" + args.site_pre + str(client_id),
            "config"
        )
        out_env_file = os.path.join(
            out_env_dir,
            "config_train.json"
        )
        
        base_env_file = out_env_file

        if os.path.isdir(out_env_dir):
            print("Modifying",base_env_file)
            with open(base_env_file) as json_file:
                env_dict = json.load(json_file)
            head, tail = os.path.split(env_dict["datalist_json_path"])
            env_dict["datalist_json_path"] = os.path.join(
                head,
                "config_brats18_datalist_client" + str(client_id-1) + ".json"
            )
            os.remove(out_env_file)

            with open(out_env_file, "w") as outfile:
                json.dump(env_dict, outfile, indent=2)

if __name__ == "__main__":
    main()
