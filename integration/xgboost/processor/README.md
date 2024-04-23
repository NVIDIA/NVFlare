# Build Instruction

This plugin build requires xgboost source code, checkout xgboost source and build it with FEDERATED plugin,

cd xgboost
mkdir build
cd build
cmake .. -DPLUGIN_FEDERATED=ON
make

cd NVFlare/integration/xgboost/processor
mkdir build
cd build
cmake ..
make
