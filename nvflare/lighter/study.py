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

from __future__ import absolute_import

import argparse
import base64
import json
import os
import time
from datetime import datetime

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from nvflare.apis.fl_constant import StudyUrn
from nvflare.lighter.utils import load_yaml


def get_root_ca(cert_file):
    persistent_state = json.load(open(cert_file, "rt"))
    pri_key = serialization.load_pem_private_key(
        persistent_state["root_pri_key"].encode("ascii"), password=None, backend=default_backend()
    )
    return pri_key


def get_input(prompt, item_list, multiple=False):
    while True:
        answer = input(prompt)
        result = None
        if multiple:
            try:
                if answer == "":
                    print("None of the choices is selected.")
                    result = []
                else:
                    trimmed = set(answer.split(","))
                    result = [item_list[int(i)] for i in trimmed]
                    print(f"{result} selected after duplicate inputs removed.")
            except BaseException:
                print("Input contains errors (non-integer or out of index range)")
        else:
            try:
                result = item_list[int(answer)]
            except ValueError:
                print(f"Expect integer but got {answer.__class__.__name__}")
            except IndexError:
                print("Number out of index range")
        if result is not None:
            break
    return result


def get_datetime_input(prompt):
    while True:
        answer = input(prompt)
        try:
            datetime_result = datetime.strptime(answer, "%m/%d/%Y %H:%M:%S")
            result = int(time.mktime(datetime_result.timetuple()))
            break
        except:
            print(f"Expect MM/DD/YYYY hh:mm:ss, but got {answer}")
    return result


def str2b64str(string):
    return base64.urlsafe_b64encode(string.encode("ascii")).decode("ascii").rstrip("=")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project_file", type=str, default="project.yml", help="file to describe FL project")
    parser.add_argument("-s", "--state_directory", type=str, help="directory with root CA info")

    args = parser.parse_args()

    current_path = os.getcwd()

    # main project file
    project_file = args.project_file
    project_full_path = os.path.join(current_path, project_file)
    if not os.path.exists(project_full_path):
        print(f"{project_full_path} not found.  Running study requires that file.")
        exit(0)

    state_dir_full_path = os.path.join(current_path, args.state_directory)
    if not os.path.exists(state_dir_full_path):
        print(f"{state_dir_full_path} not found.  Running study requires that directory.")
        exit(0)

    pv_key = get_root_ca(os.path.join(state_dir_full_path, "cert.json"))

    project = load_yaml(project_full_path)
    api_version = project.get("api_version")
    if api_version not in [3]:
        raise ValueError(f"API version expected 3 but found {api_version}")

    admin_list = list()
    client_list = list()
    for p in project.get("participants"):
        if p.get("type") == "admin":
            admin_list.append(p.get("name"))
        elif p.get("type") == "client":
            client_list.append(p.get("name"))

    admin_list_string = ", ".join([f"{i}:{v}" for i, v in enumerate(admin_list)])
    client_list_string = ", ".join([f"{i}:{v}" for i, v in enumerate(client_list)])

    name = input("Please enter the name of this study: ")
    description = input("and brief description: ")
    contact = get_input(f"select one admin for contact {admin_list_string}: ", admin_list)

    participating_admins = get_input(
        f"select participating_admins admins (separated by ',') {admin_list_string} ", admin_list, multiple=True
    )
    participating_clients = get_input(
        f"select participating clients (separated by ',') {client_list_string} ", client_list, multiple=True
    )
    # participating_clients_string = ", ".join([f"{i}:{v}" for i, v in enumerate(participating_clients)])
    # reviewer_dict = dict()
    # for admin in participating_admins:
    #     reviewed_clients = get_input(
    #         f"what clients will reviewer {admin} review {participating_clients_string} ",
    #         participating_clients,
    #         multiple=True,
    #     )
    #     reviewer_dict[admin] = reviewed_clients
    start_time = get_datetime_input("input start time of this study (MM/DD/YYYY hh:mm:ss) in UTC time: ")
    end_time = get_datetime_input("input end time of this study (MM/DD/YYYY hh:mm:ss) in UTC time: ")

    study_config = dict(
        name=name,
        description=description,
        contact=contact,
        participating_admins=participating_admins,
        participating_clients=participating_clients,
        start_time=start_time,
        end_time=end_time,
    )

    studies_config = {StudyUrn.STUDIES.value: [study_config]}
    headers = {"typ": "nvflare_study", "alg": "RS256"}
    payload = studies_config
    jws = jwt.api_jws.encode(
        payload=json.dumps(payload, separators=(",", ":")).encode("utf-8"), key=pv_key, headers=headers
    )
    with open(f"{name}.jws", "wt") as f:
        f.write(jws)
    print(f"study config file was generated at {name}.jws")


if __name__ == "__main__":
    main()
