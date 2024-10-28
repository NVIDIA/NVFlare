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

import shutil
from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm.contrib import tzip

TASK_FILE_ID = {"liver": "LVR", "spleen": "SPL", "pancreas": "PAN", "kidney": "KITS"}

TASK_LABEL_MAP = {
    "liver": {0: 0, 1: 1, 2: 2},
    "spleen": {0: 0, 1: 3},
    "pancreas": {0: 0, 1: 4, 2: 5},
    "kidney": {0: 0, 1: 6, 2: 7},
}

DEFAULT_DATA_LIST = {
    "liver": "data/Liver/datalist.json",
    "spleen": "data/Spleen/datalist.json",
    "pancreas": "data/Pancreas/datalist.json",
    "kidney": "data/KiTS19/datalist.json",
}


def map_values(data: np.ndarray, task: str) -> np.ndarray:
    data = data.astype(np.uint8)

    m = TASK_LABEL_MAP[task]
    f = np.vectorize(lambda x: m[x])

    return f(data).astype(np.uint8)


def convert_msd_dataset(src: str, dst: str, task: str) -> None:
    if not Path(src).is_dir():
        raise ValueError(f"source path {src} must be a directory.")

    images = [str(f) for f in Path(src).glob("imagesTr/*.gz")]
    assert len(images) > 0
    labels = [img.replace("imagesTr", "labelsTr") for img in images]

    Path(dst).mkdir(parents=True, exist_ok=True)
    for src_img, src_seg in tzip(images, labels):
        # Generate image file name
        dst_img = "IM_" + TASK_FILE_ID[task] + "_" + src_img.split("_")[-1]
        dst_img = str(Path(dst) / dst_img)

        # Just copy image
        shutil.copy(src_img, dst_img)

        # Generate label file name
        dst_seg = "LB_" + TASK_FILE_ID[task] + "_" + src_seg.split("_")[-1]
        dst_seg = str(Path(dst) / dst_seg)

        # Remap labels
        seg = nib.load(src_seg)
        seg_data = np.asanyarray(seg.dataobj)
        seg_data = map_values(seg_data, task=task)
        seg = nib.Nifti1Image(seg_data, seg.affine)

        # Save new label
        nib.save(seg, dst_seg)

    # Copy datalist.json to dst if necessary
    dst_list_path = str(Path(dst) / "datalist.json")
    try:
        shutil.copy(DEFAULT_DATA_LIST[task], dst_list_path)
    except shutil.SameFileError:
        pass


def convert_kits_dataset(src: str, dst: str, task: str) -> None:
    if not Path(src).is_dir():
        raise ValueError(f"source path {src} must be a directory.")

    labels = [str(f) for f in Path(src).glob("*/segmentation.nii.gz")]
    assert len(labels) > 0
    images = [f.replace("segmentation.nii", "imaging.nii") for f in labels]

    Path(dst).mkdir(parents=True, exist_ok=True)
    for src_img, src_seg in tzip(images, labels):
        case_id = Path(src_img).parent.name.replace("case_00", "")

        # Generate new file name and copy image to dst
        dst_img = str(Path(dst) / f"IM_KITS_{case_id}.nii.gz")
        shutil.copy(src_img, dst_img)

        # Generate label file name
        dst_seg = str(Path(dst) / f"LB_KITS_{case_id}.nii.gz")

        # Remap labels
        seg = nib.load(src_seg)
        seg_data = np.asanyarray(seg.dataobj)
        seg_data = map_values(seg_data, task=task)
        seg = nib.Nifti1Image(seg_data, seg.affine)

        # Save new label
        nib.save(seg, dst_seg)

    # Copy datalist.json to dst if necessary
    dst_list_path = str(Path(dst) / "datalist.json")
    try:
        shutil.copy(DEFAULT_DATA_LIST[task], dst_list_path)
    except shutil.SameFileError:
        pass


def main(args) -> None:
    if args.task in ["liver", "pancreas", "spleen"]:
        convert_msd_dataset(args.src, args.dst, args.task)
    else:
        convert_kits_dataset(args.src, args.dst, args.task)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["kidney", "liver", "pancreas", "spleen"],
        help="Choose which dataset to process.",
    )
    parser.add_argument("--src", "-s", type=str, help="Path to the dataset root directory.")
    parser.add_argument("--dst", "-d", type=str, help="Path to the output dataset directory.")
    args = parser.parse_args()

    main(args)
