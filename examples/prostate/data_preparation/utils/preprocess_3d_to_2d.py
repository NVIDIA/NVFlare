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
import glob
import os
import pathlib

import nibabel as nib
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Convert 3D prostate dataset for 2D experiment")
    parser.add_argument("--data_dir", type=str, help="Path to all data folder")
    parser.add_argument("--site_name", type=str, help="Path to particular set")
    parser.add_argument("--out_path", type=str, help="Path to output 2D file folder")
    args = parser.parse_args()
    # output folder main path
    image_out_path = os.path.join(args.out_path, args.site_name, "Image")
    mask_out_path = os.path.join(args.out_path, args.site_name, "Mask")
    # get input files
    image_file_path = os.path.join(args.data_dir, args.site_name, "Image", "*")
    mask_file_path = os.path.join(args.data_dir, args.site_name, "Mask", "*")
    image_files = glob.glob(image_file_path)
    mask_files = glob.glob(mask_file_path)
    assert len(image_files) == len(mask_files), "The number of image and mask files should be the same."
    # iterate through input files and convert 3D dataset to 2D
    for idx in range(len(image_files)):
        # collect the paths
        image_file_name = image_files[idx]
        mask_file_name = mask_files[idx]
        # load image and mask
        image_case_id = os.path.basename(image_file_name)
        mask_case_id = os.path.basename(image_file_name)
        assert image_case_id == mask_case_id, "Image and mask ID should match."
        case_id = image_case_id.replace(".nii.gz", "")
        # read nii.gz files with nibabel
        image = nib.load(image_file_name).get_fdata().transpose((1, 0, 2))
        mask = nib.load(mask_file_name).get_fdata().transpose((1, 0, 2))
        # clip and normalize image
        image = (image - 0) / (2048 - 0)
        image[image > 1] = 1
        # iterate through slice dimension
        for slice_idx in range(image.shape[2]):
            image_slice = image[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx]
            # only extract slices with mask annotation
            if np.sum(mask_slice) > 0:
                # scale to 0~255
                image_slice = image_slice * 255
                mask_slice = mask_slice * 255
                # output path
                pathlib.Path(os.path.join(image_out_path, case_id)).mkdir(parents=True, exist_ok=True)
                pathlib.Path(os.path.join(mask_out_path, case_id)).mkdir(parents=True, exist_ok=True)
                # flip so as to follow clinical viewing orientation
                im = Image.fromarray(image_slice).convert("L").transpose(Image.FLIP_TOP_BOTTOM)
                im.save(os.path.join(image_out_path, case_id, "{}.png".format(slice_idx)))
                im = Image.fromarray(mask_slice).convert("L").transpose(Image.FLIP_TOP_BOTTOM)
                im.save(os.path.join(mask_out_path, case_id, "{}.png".format(slice_idx)))
    return


if __name__ == "__main__":
    main()
