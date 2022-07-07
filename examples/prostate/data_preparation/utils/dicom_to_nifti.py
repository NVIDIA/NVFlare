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

import dicom2nifti

parser = argparse.ArgumentParser("Dicom to Nifti converter")
parser.add_argument("--dicom_folder", help="Input Dicom folder path", type=str)
parser.add_argument("--nifti_path", help="Output Nifti file path", type=str)
args = parser.parse_args()

dicom2nifti.dicom_series_to_nifti(args.dicom_folder, args.nifti_path)
