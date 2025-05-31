#!/bin/bash
set -e

# Check exact number of arguments
if [ $# -ne 3 ]; then
  echo "Usage: $0 <yaml_config> <site_name> <startup_folder>"
  exit 1
fi

CVM_ID=$(head -c 6 /dev/urandom | xxd -p)

echo "Building CVM with ID: $CVM_ID"

YAML_CONFIG=$1
SITE_NAME=$2
STARTUP_FOLDER=$3

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_DIR}/base_images/nvidia_cc_base.qcow2"
TARGET_IMAGE="${BASE_DIR}/target/${CVM_ID}_${SITE_NAME}_cvm.qcow2"
APPLOG_BASE="${BASE_DIR}/base_images/applog_base.qcow2"
APPLOG_IMAGE="${BASE_DIR}/target/${CVM_ID}_${SITE_NAME}_applog.qcow2"

echo "Copy images ..."

mkdir -p "${BASE_DIR}/target"
cp $BASE_IMAGE $TARGET_IMAGE
cp $APPLOG_BASE $APPLOG_IMAGE

echo "Start VM ..."

nohup sudo base_images/launch_vm.sh $TARGET_IMAGE $APPLOG_IMAGE > cvm.log 2>&1 &

# Wait for SSH to be ready
echo "Waiting for CVM to become reachable..."
while ! nc -z -w 1 localhost 2222; do
  sleep 1
done

echo "Wait a few minutes for ssh to come up ..."
sleep 60 

echo "Starting playbook ..."

echo "Playbook: $BASE_DIR/playbooks/cvm_build.yml"
# Run ansible playbook to configure the VM
ansible-playbook -i $BASE_DIR/playbooks/inventory.ini $BASE_DIR/playbooks/cvm_build.yml \
    -e "yaml_config=$YAML_CONFIG site_name=$SITE_NAME startup_folder=$STARTUP_FOLDER 
        cvm_id=$CVM_ID target_image=$TARGET_IMAGE applog_image=$APPLOG_IMAGE"

echo "CVM ${TARGET_IMAGE} build completed"
