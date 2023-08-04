# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import requests


def download_dir_from_github(repo_owner, repo_name, branch_name, directory_path, download_path, github_token):
    base_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{directory_path}?ref={branch_name}'
    headers = {'Accept': 'application/vnd.github.v3+json',
               'Authorization': f'token {github_token}'
               }

    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching directory contents: {e}")
        return

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for item in response.json():
        if 'type' in item:
            if item['type'] == 'file':
                file_name = item['name']
                file_path = os.path.join(download_path, item['path'])
                file_parent = os.path.dirname(file_path)

                try:
                    if not os.path.exists(file_parent):
                        os.makedirs(file_parent)

                    download_url = item['download_url']
                    if download_url and str(download_url) != 'None':
                        file_response = requests.get(download_url)
                        file_response.raise_for_status()
                        with open(file_path, 'wb') as f:
                            f.write(file_response.content)
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading file {file_name}: {e}")
            elif item['type'] == 'dir':
                subdir_path = os.path.join(download_path, item['name'])
                print("download", subdir_path)
                download_dir_from_github(repo_owner, repo_name, branch_name, item['path'], subdir_path, github_token)




def main():
    import sys
    print(sys.argv)
    download_dir_from_github(repo_owner="NVIDIA",
                             repo_name="NVFlare",
                             branch_name="dev",
                             directory_path="examples/hello-world/ml-to-fl",
                             download_path="/tmp/nvflare/downloads/template",
                             github_token="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")


if __name__ == "__main__":
    main()
