# XGBoost plugins


## Install required dependencies for CUDA plugin
If you want to build CUDA plugin, you need to install the following libraries:
Require `libgmp-dev`, CUDA runtime >= 12.1, CUDA driver >= 12.1, NVIDIA GPU Driver >= 535
Compute Compatibility >= 7.0

## Build instructions

```
mkdir build
cd build
cmake ..
make
```

## Disable build of CUDA plugin
You can pass option to cmake to disable the build of CUDA plugin if you don't have the environment:
```
cmake -DBUILD_CUDA_PLUGIN=OFF ..
```

