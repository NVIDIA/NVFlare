# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

FROM python:3.12-slim

# Supply the exact NVFlare wheel and any dependency wheels approved for deployment.
COPY wheels/ /opt/nvflare-wheels/
RUN python3 -m pip install --no-cache-dir /opt/nvflare-wheels/*.whl

# The administrator's public key is fixed before generating the Azure policy.
# Never copy the private signing key, final startup kit, or policy into this image.
COPY deployment-public-key.pem /opt/nvflare-trust/deployment-public-key.pem
RUN chmod 0444 /opt/nvflare-trust/deployment-public-key.pem
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/python3", "-I", "-m", "nvflare.app_opt.confidential_computing.aci_bootstrap"]
