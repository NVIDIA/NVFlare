# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Constants for file transfer command module."""

SERVER_MODULE_NAME = "file_transfer"
SERVER_CMD_UPLOAD_TEXT = "_upload_text_file"
SERVER_CMD_DOWNLOAD_TEXT = "_download_text_file"
SERVER_CMD_UPLOAD_BINARY = "_upload_binary_file"
SERVER_CMD_DOWNLOAD_BINARY = "_download_binary_file"
SERVER_CMD_UPLOAD_FOLDER = "_upload_folder"
SERVER_CMD_SUBMIT_JOB = "_submit_job"
SERVER_CMD_DOWNLOAD_JOB = "_download_job"
SERVER_CMD_INFO = "_info"
SERVER_CMD_PULL_BINARY = "_pull_binary_file"


DOWNLOAD_URL_MARKER = "Download_URL:"
PUSH_FOLDER_FQN = "file_transfer.push_folder"
DOWNLOAD_FOLDER_FQN = "file_transfer.download_folder"
PULL_FOLDER_FQN = "file_transfer.pull_folder"
PULL_BINARY_FQN = "file_transfer.pull_binary"
