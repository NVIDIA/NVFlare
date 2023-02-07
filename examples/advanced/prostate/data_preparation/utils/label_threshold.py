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

import nibabel as nib

parser = argparse.ArgumentParser("Threshold label image to a binary one")
parser.add_argument("--input_path", help="Input label image path", type=str)
parser.add_argument("--output_path", help="Output binary image path", type=str)
parser.add_argument("--threshold", help="threshold", type=int, default=0)
args = parser.parse_args()

img = nib.load(args.input_path)
img_np = img.get_fdata()
img_affine = img.affine
img_np[img_np > args.threshold] = 1

nft_img = nib.Nifti1Image(img_np, img_affine)
nib.save(nft_img, args.output_path)
