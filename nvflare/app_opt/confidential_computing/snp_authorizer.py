# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import jwt
from jwt import PyJWKClient
import subprocess
import base64
import traceback
import os

from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer

SNP_NAMESPACE = "x-snp"

class SNPAuthorizer(CCAuthorizer):
    def generate(self):
        cmd = ['sudo', 'snpguest', 'report', 'report.bin', 'request.bin']
        with open('request.bin', 'wb') as request_file:
            request_file.write(b'\x01'*64)
        cp = subprocess.run(cmd, capture_output=True)
        # print(token)
        with open('report.bin', 'rb') as report_file:
            token = base64.b64encode(report_file.read())
        # print(token)
        return token
    
    def verify(self, token):
        # print("Now verifying SNP")
        try:
            # if not os.path.exists('./cert'):
            #     os.mkdir('./cert')
            # cmd = ['snpguest', 'fetch', 'ca', 'der', 'genoa', './cert', '--endorser', 'vcek']
            # cp = subprocess.run(cmd, capture_output=True)
            # if cp.returncode != 0:
            #     return False
            report_bin = base64.b64decode(token)
            with open('rcv_report.bin', 'wb') as report_file:
                report_file.write(report_bin)
            # cmd = ['snpguest', 'fetch', 'vcek', 'der', 'genoa', './cert', 'rcv_report.bin']
            # cp = subprocess.run(cmd, capture_output=True)
            # if cp.returncode != 0:
            #     return False
            # cmd = ['snpguest', 'verify', 'certs', './cert']
            # cp = subprocess.run(cmd, capture_output=True)
            # if cp.returncode != 0:
            #     return False
            cmd = ['snpguest', 'verify', 'attestation', './cert', 'rcv_report.bin']
            cp = subprocess.run(cmd, capture_output=True)
            if cp.returncode != 0:
                return False
            print(f"{cp.stdout=}\n{cp.stderr=} Now returning True")
            # print(f"{claims=}")
<<<<<<< HEAD
            print("Now returning True")
=======
>>>>>>> 3bd88414 (Fix logger)
            return True
        except:
            print(traceback.format_exc())
            return False
        return True

    def can_generate(self) -> bool:
        return True

    def can_verify(self) -> bool:
        return True

    def get_namespace(self) -> str:
        return SNP_NAMESPACE


if __name__ == "__main__":
  m = SNPAuthorizer()
  token = m.generate()
  print(type(token))
  v = m.verify(token)
  print(v)

