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
import nrrd
import numpy as np

parser = argparse.ArgumentParser("Convert nrrd label to nifti with reference image file for affine")
parser.add_argument("--input_path", help="Input nrrd path", type=str)
parser.add_argument("--reference_path", help="Reference image path", type=str)
parser.add_argument("--output_path", help="Output nifti path", type=str)
args = parser.parse_args()

img = nib.load(args.reference_path)
img_affine = img.affine

nrrd = nrrd.read(args.input_path)
data = np.flip(nrrd[0], axis=1)

nft_img = nib.Nifti1Image(data, img_affine)
nib.save(nft_img, args.output_path)
