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

import argparse
import base64
import os

import tenseal as ts


def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate HE context")
    parser.add_argument("--scheme", type=str, default="BFV", help="HE scheme, default is BFV")
    parser.add_argument("--poly_modulus_degree", type=int, default=4096, help="Poly modulus degree, default is 4096")
    parser.add_argument("--out_path", type=str, help="Output root path for HE context files for client and server")
    return parser


def write_data(file_name: str, data: bytes):
    data = base64.b64encode(data)
    with open(file_name, "wb") as f:
        f.write(data)


def main():
    parser = data_split_args_parser()
    args = parser.parse_args()
    if args.scheme == "BFV":
        scheme = ts.SCHEME_TYPE.BFV
        # Generate HE context
        context = ts.context(scheme, poly_modulus_degree=args.poly_modulus_degree, plain_modulus=1032193)
    elif args.scheme == "CKKS":
        scheme = ts.SCHEME_TYPE.CKKS
        # Generate HE context, CKKS does not need plain_modulus
        context = ts.context(scheme, poly_modulus_degree=args.poly_modulus_degree)
    else:
        raise ValueError("HE scheme not supported")

    # Save HE context to file for client
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    context_serial = context.serialize(save_secret_key=True)
    write_data(os.path.join(args.out_path, "he_context_client.txt"), context_serial)

    # Save HE context to file for server
    context_serial = context.serialize(save_secret_key=False)
    write_data(os.path.join(args.out_path, "he_context_server.txt"), context_serial)


if __name__ == "__main__":
    main()
