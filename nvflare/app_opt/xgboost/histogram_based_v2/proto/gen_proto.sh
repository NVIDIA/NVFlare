#!/usr/bin/env sh
# Install grpcio-tools:
#   pip install grpcio-tools
# or
#   mamba install grpcio-tools
python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. federated.proto
