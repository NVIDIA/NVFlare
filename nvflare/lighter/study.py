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
import json
import os
from datetime import datetime

import yaml


def get_input(prompt, item_list, multiple=False):
    answer = input(prompt)
    result = None
    if multiple:
        try:
            if answer == "":
                result = []
            else:
                result = [item_list[int(i)] for i in answer.split(",")]
        except BaseException:
            print("Input contains errors (non-integer or out of index range)")
    else:
        try:
            result = item_list[int(answer)]
        except ValueError:
            print(f"Expect integer but got {answer.__class__.__name__}")
        except IndexError:
            print("Number out of index range")
    if result is None:
        exit(1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project_file", type=str, default="project.yml", help="file to describe FL project")

    args = parser.parse_args()

    current_path = os.getcwd()

    # main project file
    project_file = args.project_file
    project_full_path = os.path.join(current_path, project_file)
    if not os.path.exists(project_full_path):
        print(f"{project_full_path} not found.  Running study requires that file.")
        exit(0)

    project = yaml.load(open(project_full_path, "r"), Loader=yaml.Loader)
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
    study_name = input("Please enter the name of this study: ")
    study_description = input("and brief description: ")
    contact = get_input(f"select one admin for contact {admin_list_string}: ", admin_list)
    participanting_admins = get_input(
        f"select participanting admins (separated by ',') {admin_list_string} ", admin_list, multiple=True
    )
    participanting_clients = get_input(
        f"select participanting clients (separated by ',') {client_list_string} ", client_list, multiple=True
    )
    participanting_clients_string = ", ".join([f"{i}:{v}" for i, v in enumerate(participanting_clients)])
    reviewer_dict = dict()
    for admin in participanting_admins:
        reviewed_clients = get_input(
            f"what clients will reviewer {admin} review {participanting_clients_string} ",
            participanting_clients,
            multiple=True,
        )
        reviewer_dict[admin] = reviewed_clients
    start_date = input("input start date of this study (MM/DD/YYYY): ")
    try:
        start_date = datetime.strptime(start_date, "%m/%d/%Y").date().isoformat()
    except:
        print(f"Expect MM/DD/YYYY but got {start_date}")
        exit(1)
    end_date = input("input end date of this study (MM/DD/YYYY): ")
    try:
        end_date = datetime.strptime(end_date, "%m/%d/%Y").date().isoformat()
    except:
        print(f"Expect MM/DD/YYYY but got {end_date}")
        exit(1)

    study_config = dict(
        study_name=study_name,
        description=study_description,
        contact=contact,
        participanting_admins=participanting_admins,
        participanting_clients=participanting_clients,
        reviewers=reviewer_dict,
        start_date=start_date,
        end_date=end_date,
    )
    with open(f"{study_name}.json", "wt") as f:
        f.write(json.dumps(study_config, indent=2))


if __name__ == "__main__":
    main()
