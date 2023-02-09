# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import ssl

from application import init_app

app = init_app()

if __name__ == "__main__":
    web_root = os.environ.get("NVFL_WEB_ROOT", "/var/tmp/nvflare/dashboard")
    web_crt = os.path.join(web_root, "cert", "web.crt")
    web_key = os.path.join(web_root, "cert", "web.key")
    port = os.environ.get("NVFL_WEB_PORT", "8443")
    if os.path.exists(web_crt) and os.path.exists(web_key):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(web_crt, web_key)
    else:
        ssl_context = None
    app.run(host="0.0.0.0", port=port, ssl_context=ssl_context)
