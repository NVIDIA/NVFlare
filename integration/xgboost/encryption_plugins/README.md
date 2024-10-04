# XGBoost Encryption Plugins

## Install required dependencies for building encryption plugins

Following libraries are required:
* gcc
* CMake>=3.19
* libgmp3-dev
* CUDA Driver and runtime 12.2 or 12.4
* NVIDIA GPU Driver >= 535 Compute Capability >= 7.0.

On the building site:    
```bash
    git clone https://github.com/NVIDIA/NVFlare.git
    cd NVFlare/integration/xgboost/encryption_plugins
    git submodule update --init --recursive
```

## Building Plugins
Under `integration/xgboost/encryption_plugins`, run the build commands
```bash
    mkdir build
    cd build
    cmake ..
    make
```
The generated plugin files under build folder are,
```
    cuda_plugin/libcuda_paillier.so
    nvflare_plugin/libnvflare.so
```

> **_NOTE:_**  You can pass option to cmake to disable the build of CUDA plugin
> if you don't have the environment: ```cmake -DBUILD_CUDA_PLUGIN=OFF ..```


## How to run XGBoost with encryption plugins
For each client site:

1. Copy the pre-built ".so" files to each site
2. Make sure you have installed required dependencies: GPU Driver >= 535, CUDA Driver and runtime 12.2 or 12.4, `libgmp3-dev`
3. Update the site local resource.json to point to the appropriate ".so" file

For handling these complex environment, we recommend you build a docker image so every
client site can just use it.
