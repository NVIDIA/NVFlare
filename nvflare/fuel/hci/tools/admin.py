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

from nvflare.fuel.hci.client.cli import AdminClient, CredentialType
from nvflare.fuel.hci.client.file_transfer import FileTransferModule


def main():
    """
    Script to launch the admin client to issue admin commands to the server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=55550)
    parser.add_argument("--prompt", type=str, default="> ")
    parser.add_argument("--with_file_transfer", action="store_true")
    parser.add_argument("--upload_folder_cmd_name", type=str, default="upload_app")
    parser.add_argument("--upload_dir", type=str, default="")
    parser.add_argument("--download_dir", type=str, default="")
    parser.add_argument("--with_shell", action="store_true")
    parser.add_argument("--with_login", action="store_true")
    parser.add_argument("--cred_type", default="password")
    parser.add_argument("--with_ssl", action="store_true")
    parser.add_argument("--ca_cert", type=str, default="")
    parser.add_argument("--client_cert", type=str, default="")
    parser.add_argument("--client_key", type=str, default="")
    parser.add_argument("--with_debug", action="store_true")

    args = parser.parse_args()

    modules = []

    if args.with_file_transfer:
        modules.append(
            FileTransferModule(
                upload_dir=args.upload_dir,
                download_dir=args.download_dir,
                upload_folder_cmd_name=args.upload_folder_cmd_name,
            )
        )

    ca_cert = args.ca_cert
    client_cert = args.client_cert
    client_key = args.client_key

    if args.with_ssl:
        if len(ca_cert) <= 0:
            print("missing CA Cert file name")
            return

        if len(client_cert) <= 0:
            print("missing Client Cert file name")
            return

        if len(client_key) <= 0:
            print("missing Client Key file name")
            return
    else:
        ca_cert = None
        client_key = None
        client_cert = None

    if args.with_debug:
        print("SSL: {}".format(args.with_ssl))
        print("User Login: {}".format(args.with_login))
        print("File Transfer: {}".format(args.with_file_transfer))

        if args.with_file_transfer:
            print("  Upload Dir: {}".format(args.upload_dir))
            print("  Download Dir: {}".format(args.download_dir))

    print("Admin Server: {} on port {}".format(args.host, args.port))

    client = AdminClient(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        cmd_modules=modules,
        ca_cert=ca_cert,
        client_cert=client_cert,
        client_key=client_key,
        require_login=args.with_login,
        credential_type=CredentialType.PASSWORD if args.cred_type == "password" else CredentialType.CERT,
        debug=args.with_debug,
    )

    client.run()


if __name__ == "__main__":
    main()
