# XGBoost plugins


## Install required dependencies for building CUDA plugin
If you want to build the CUDA plugin on your own, you need to install the following libraries:
Require `libgmp3-dev`, CMake>=3.19, CUDA runtime >= 12.1, CUDA driver >= 12.1, NVIDIA GPU Driver >= 535
Compute Compatibility >= 7.0

On the building site:
0. Install GPU Driver >= 535, CUDA runtime >= 12.1, CUDA driver >= 12.1
1. Install `libgmp3-dev`, gcc, CMake
2. Clone the NVFlare main branch and update the submodule
    ```
    git clone https://github.com/NVIDIA/NVFlare.git \
    && cd NVFlare/integration/xgboost/encryption_plugins \
    && git submodule update --init --recursive
    ```
3. Under integration/xgboost/encryption_plugins, run the build commands
    ```
    mkdir build
    cd build
    cmake ..
    make
    ```

> **_NOTE:_**  You can pass option to cmake to disable the build of CUDA plugin
> if you don't have the environment: ```cmake -DBUILD_CUDA_PLUGIN=OFF ..```


## How to run with CUDA plugin
For each client site:

0. Copy the pre-built ".so" file to each site
1. Make sure you have installed required dependencies: GPU Driver >= 535, CUDA runtime >= 12.1, CUDA driver >= 12.1, `libgmp3-dev`
2. Update the site local resource.json to point to the ".so" file

For handling these complex environment, we recommend you build a docker image so every
client site can just use it.
