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

def result_post(body=None):  # noqa: E501
    """Report result for the task

     # noqa: E501

    :param body:
    :type body: dict | bytes

    :rtype: ResultResponse
    """
    if connexion.request.is_json:
        body = TaskResult.from_dict(connexion.request.get_json())  # noqa: E501
    return "do some magic!"


def study_post(body=None):  # noqa: E501
    """Get the study

     # noqa: E501

    :param body:
    :type body: dict | bytes

    :rtype: StudyResponse
    """
    if connexion.request.is_json:
        body = StudyRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return "do some magic!"


def task_post(body=None):  # noqa: E501
    """Get a task

     # noqa: E501

    :param body:
    :type body: dict | bytes

    :rtype: TaskResponse
    """
    if connexion.request.is_json:
        body = TaskRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return "do some magic!"
