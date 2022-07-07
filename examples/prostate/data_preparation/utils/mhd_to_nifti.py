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

import argparse

import numpy as np
import SimpleITK as sitk

parser = argparse.ArgumentParser("Convert mhd file to nifti")
parser.add_argument("--input_path", help="Input mhd path", type=str)
parser.add_argument("--output_path", help="Output nifti path", type=str)
args = parser.parse_args()

reader = sitk.ImageFileReader()
reader.SetImageIO("MetaImageIO")
reader.SetFileName(args.input_path)
image = reader.Execute()

writer = sitk.ImageFileWriter()
writer.SetFileName(args.output_path)
writer.Execute(image)
