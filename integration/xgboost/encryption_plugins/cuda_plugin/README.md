# CUDA plugin

Use CUDA to do paillier encryption and addition.

# Build Instruction

## Install required dependencies
Require `libgmp-dev`, CUDA runtime >= 12.1, CUDA driver >= 12.1, NVIDIA GPU Driver >= 535
Compute Compatibility >= 7.0

## Build libproc_cuda_paillier.so
```
mkdir build
cd build
cmake ..
make
```
