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
import pytest

# temp disable import
# from nvflare.app_opt.psi.dh_psi.dh_psi_client import PSIClient
# from nvflare.app_opt.psi.dh_psi.dh_psi_server import PSIServer
#


class TestPSIAlgo:
    # Comment out the PSI tests for now.
    @pytest.mark.parametrize(
        "test_input,  expected",
        [
            (
                {
                    "server_items": [
                        "user_id-100",
                        "user_id-106",
                        "user_id-112",
                        "user_id-118",
                        "user_id-124",
                        "user_id-130",
                        "user_id-136",
                        "user_id-142",
                        "user_id-148",
                        "user_id-154",
                        "user_id-160",
                        "user_id-166",
                        "user_id-172",
                        "user_id-178",
                        "user_id-184",
                        "user_id-190",
                        "user_id-196",
                    ],
                    "client_items": [
                        "user_id-100",
                        "user_id-104",
                        "user_id-108",
                        "user_id-112",
                        "user_id-116",
                        "user_id-120",
                        "user_id-124",
                        "user_id-128",
                        "user_id-132",
                        "user_id-136",
                        "user_id-140",
                        "user_id-144",
                        "user_id-148",
                        "user_id-152",
                        "user_id-156",
                        "user_id-160",
                        "user_id-164",
                        "user_id-168",
                        "user_id-172",
                        "user_id-176",
                        "user_id-180",
                        "user_id-184",
                        "user_id-188",
                        "user_id-192",
                        "user_id-196",
                        "user_id-200",
                        "user_id-204",
                        "user_id-208",
                        "user_id-212",
                        "user_id-216",
                        "user_id-220",
                        "user_id-224",
                        "user_id-228",
                        "user_id-232",
                        "user_id-236",
                        "user_id-240",
                        "user_id-244",
                        "user_id-248",
                        "user_id-252",
                        "user_id-256",
                        "user_id-260",
                        "user_id-264",
                        "user_id-268",
                        "user_id-272",
                        "user_id-276",
                        "user_id-280",
                        "user_id-284",
                        "user_id-288",
                        "user_id-292",
                        "user_id-296",
                    ],
                },
                [
                    "user_id-100",
                    "user_id-112",
                    "user_id-124",
                    "user_id-136",
                    "user_id-148",
                    "user_id-160",
                    "user_id-172",
                    "user_id-184",
                    "user_id-196",
                ],
            ),
        ],
    )
    def test_psi_algo(self, test_input, expected):
        # have to comment out the unittests for now until we figure
        # out how to enable unit tests for optional requirements
        # if you want to run the test, just uncomment the following code

        # temp disable tests as Jenkins machine is based on Ubuntu 18.04 and missing
        # ImportError: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.29'
        # not found (required by /root/.local/share/virtualenvs/NVFlare-premerge-R2yT5_j2/lib/python3.8/site-packages/private_set_intersection/python/_openmined_psi.so)
        #
        # server_items = test_input["server_items"]
        # client_items = test_input["client_items"]
        # client = PSIClient(client_items)
        # server = PSIServer(server_items)
        # setup_msg = server.setup(len(client_items))
        #
        # client.receive_setup(setup_msg)
        # request_msg = client.get_request(client_items)
        # response_msg = server.process_request(request_msg)
        # intersections = client.get_intersection(response_msg)
        #
        # assert 9 == len(intersections)
        # assert intersections == expected

        pass
