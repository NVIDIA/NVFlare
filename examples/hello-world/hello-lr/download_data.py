# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


def download_file(url: str, outdir: str):
    # make sure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # get filename from URL
    filename = url.split("/")[-1]
    filepath = os.path.join(outdir, filename)

    # download in streaming mode
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Downloaded: {filepath}")
    return filepath

# Example usage:
url = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
outdir = "/tmp/flare/dataset/heart_disease_data"
download_file(url, outdir)

