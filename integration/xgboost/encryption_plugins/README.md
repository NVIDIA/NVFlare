# XGBoost Encryption Plugins


## Install required dependencies for building CUDA plugin
If you want to build the CUDA plugin, you need to install the following libraries:
Require `libgmp3-dev`, CMake>=3.19, CUDA Driver and runtime 12.2 or 12.4, NVIDIA GPU Driver >= 535
Compute Compatibility >= 7.0.

On the building site:
1. Install GPU Driver >= 535, CUDA Driver and runtime 12.2 or 12.4
2. Install `libgmp3-dev`, gcc, CMake
3. Clone the NVFlare main branch and update the submodule
    ```
    git clone https://github.com/NVIDIA/NVFlare.git \
    && cd NVFlare/integration/xgboost/encryption_plugins \
    && git submodule update --init --recursive
    ```

## Building Plugins
Under integration/xgboost/encryption_plugins, run the build commands
    ```
    mkdir build
    cd build
    cmake ..
    make
    ```
The generated plugin files under build folder are,
    ```
    cuda_pluign/libcuda_paillier.so
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
