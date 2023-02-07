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
import numpy as np

parser = argparse.ArgumentParser("Combine label images to a binary one")
parser.add_argument(
    "--ref_image", help="Reference image file path, to make sure algnment between image and mask", type=str
)
parser.add_argument("--input_folder_path", help="Input label image folder path", type=str)
parser.add_argument("--output_path", help="Output binary image path", type=str)
args = parser.parse_args()

ref = nib.load(args.ref_image)
ref_affine = ref.affine
ref_np = ref.get_fdata()

img = nib.load(args.input_folder_path + "/1.nii.gz")
img_np = img.get_fdata()
img = nib.load(args.input_folder_path + "/2.nii.gz")
img_np = img_np + img.get_fdata()
img = nib.load(args.input_folder_path + "/4.nii.gz")
img_np = img_np + img.get_fdata()
# Special treatment for urethra: if urethra only, then discard
# since it is usually not included in other prostate segmentation protocols
ure = nib.load(args.input_folder_path + "/3.nii.gz")
ure_np = ure.get_fdata()
for slice_idx in range(img_np.shape[2]):
    image_slice = img_np[:, :, slice_idx]
    ure_slice = ure_np[:, :, slice_idx]
    if np.sum(image_slice) > 0:
        image_slice = image_slice + ure_slice
    img_np[:, :, slice_idx] = image_slice

img_np[img_np > 0] = 1
img_affine = img.affine

# reorient mask image
img_ornt = nib.io_orientation(img_affine)
ref_ornt = nib.io_orientation(ref_affine)
spatial_ornt = nib.orientations.ornt_transform(img_ornt, ref_ornt)
img_np = nib.orientations.apply_orientation(img_np, spatial_ornt)

# resample mask image
img = nib.Nifti1Image(img_np, ref_affine)
nib.save(img, args.output_path)
