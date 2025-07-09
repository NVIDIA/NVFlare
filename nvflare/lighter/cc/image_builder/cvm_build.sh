#!/bin/bash
set -e
# set -x

start=$(date +%s)

# Check exact number of arguments
if [ $# -ne 3 ]; then
  echo "Usage: $0 <yaml_config> <site_name> <startup_folder>"
  exit 1
fi

CVM_ID=$(head -c 6 /dev/urandom | xxd -p)
ENC_KEY=$(head -c 64 /dev/urandom | xxd -c 256 -p)
RANDOM_PW=$(head -c 16 /dev/urandom | base64)

echo "Building CVM with ID: $CVM_ID"

YAML_CONFIG=$1
SITE_NAME=$2
STARTUP_FOLDER=$3

SITE_STARTUP_KIT="${STARTUP_FOLDER%/}/${SITE_NAME}"

if [ ! -d "${SITE_STARTUP_KIT}" ]; then
  echo "Site startup folder $SITE_STARTUP_KIT does not exist"
  exit 1
fi

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE_FOLDER="${BASE_DIR}/base_images"
TARGET_FOLDER="${BASE_DIR}/target/cvm_${CVM_ID}"
SCRATCH_FOLDER="${BASE_DIR}/scratch/cvm_${CVM_ID}"

mkdir -p "${TARGET_FOLDER}"
mkdir -p "${SCRATCH_FOLDER}"

KEY_FILE="${SCRATCH_FOLDER}/key.txt"
PW_FILE="${SCRATCH_FOLDER}/pw.txt"

echo "$ENC_KEY" > "$KEY_FILE"
echo "$RANDOM_PW" > "$PW_FILE"


echo "Copy base images ..."

# Copy files that don't change
for f in applog.qcow2 bzImage initramfs.cpio.gz OVMF_AMD.fd README.txt; do
  cp "$BASE_IMAGE_FOLDER/$f" $TARGET_FOLDER
done

# Copy files used for building the CVM to scratch folder
cp  $BASE_IMAGE_FOLDER/rootfs.qcow2 $SCRATCH_FOLDER

export CVM_ID
envsubst < $BASE_DIR/scripts/launch_vm.sh > $TARGET_FOLDER/launch_vm.sh
envsubst < $BASE_DIR/scripts/launch_vm_with_log.sh > $SCRATCH_FOLDER/launch_vm_with_log.sh

chmod a+x $TARGET_FOLDER/launch_vm.sh
chmod a+x $SCRATCH_FOLDER/launch_vm_with_log.sh

echo "Start clear VM ..."

sudo pkill -if qemu-system-x86_64 || true
sleep 5

LOG_FILE=$SCRATCH_FOLDER/boot.log
touch $LOG_FILE

cd $TARGET_FOLDER
nohup $BASE_DIR/scripts/launch_non_cc_vm.sh $SCRATCH_FOLDER/rootfs.qcow2 \
    $TARGET_FOLDER/applog.qcow2 $TARGET_FOLDER/user_data.qcow2 $LOG_FILE > $LOG_FILE  2>&1 &

echo "Waiting for VM to get ready ..."
while ! grep -q " login:" "$LOG_FILE"; do
   sleep 1
done

echo "VM is ready"

echo "Starting playbook ..."

# Run ansible playbook to configure the VM
ansible-playbook -i $BASE_DIR/playbooks/inventory.ini $BASE_DIR/playbooks/cvm_build.yml \
    -e "yaml_config=$YAML_CONFIG site_name=$SITE_NAME startup_folder=$STARTUP_FOLDER 
         cvm_id=$CVM_ID target_folder=$TARGET_FOLDER scratch_folder=$SCRATCH_FOLDER"

echo "Encrypting the root filesystem ..."
$BASE_DIR/scripts/create_drive.sh $TARGET_FOLDER/crypt_root.qcow2 $SCRATCH_FOLDER/rootfs.qcow2 $KEY_FILE

CC_LOG_FILE=$SCRATCH_FOLDER/cc_boot.log
touch $CC_LOG_FILE

sudo pkill -if -9 qemu-system-x86_64 > /dev/null 2>&1|| true
stty sane
echo "Starting VM in CC mode"

cd $TARGET_FOLDER
nohup $SCRATCH_FOLDER/launch_vm_with_log.sh $CC_LOG_FILE > $CC_LOG_FILE  2>&1 &

echo "Waiting for VM to fail in initramfs ..."
while ! grep -q "Measurement: " "$CC_LOG_FILE"; do
   sleep 1
done

sudo pkill -if -9 qemu-system-x86_64 > /dev/null 2>&1|| true
stty sane

HASH=$(sed -n 's/.*Measurement: \(.*\)/\1/p' $CC_LOG_FILE)
HASH=$(echo $HASH | tr -d '\r')

# Register the encryption key
echo "Registering the encryption key with KBS ..."
$BASE_DIR/scripts/register_key.sh $CVM_ID $KEY_FILE

# Add measurement to the policy file
echo "Updating the policy file with the measurement $HASH..."
$BASE_DIR/scripts/update_policy.sh $HASH


echo "CVM Bundle ${TARGET_FOLDER} is ready"

end=$(date +%s)
elapsed=$((end - start))
echo "Build time: $elapsed seconds"