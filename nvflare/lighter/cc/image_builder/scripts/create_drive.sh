#!/bin/bash
set -e
# set -x

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <target_drive> <source_drive> <key_file>"
    exit 1
fi

ENC_DISK="$1"
SOURCE_DISK="$2"
ENC_KEY="$3"

# Close any open luks device
for mapper in /dev/mapper/*; do
    filename="${mapper##*/}"
    if [ "$filename" != "control" ]; then
        echo "Closing $filename"
        sudo cryptsetup luksClose $filename || true
    fi
done


TOTAL_SIZE_GB=10
# Ensure minimum size of 1G
if [ "$TOTAL_SIZE_GB" -lt 1 ]; then
    TOTAL_SIZE_GB=1
fi

# Set image size
IMAGESIZE="${TOTAL_SIZE_GB}G"

# Generate a random mapper name
MAPPER_NAME="crypt_root"
MOUNTPOINT="/mnt/$MAPPER_NAME"
NBD_DEVICE="/dev/nbd1"

# Ensure nbd kernel module is loaded
echo "[*] Loading nbd module..."
sudo modprobe nbd max_part=8

# Create qcow2 image
echo "[*] Creating qcow2 disk image '$ENC_DISK' of size $IMAGESIZE..."
qemu-img create -f qcow2 "$ENC_DISK" "$IMAGESIZE"

# Connect image to nbd device
echo "[*] Connecting qcow2 to $NBD_DEVICE..."
sudo qemu-nbd --connect="$NBD_DEVICE" "$ENC_DISK"

# Wait for /dev/nbd0 to be ready
sleep 1

# Format with LUKS
echo "[*] Formatting LUKS..."
sudo cryptsetup luksFormat "$NBD_DEVICE" \
    --batch-mode         \
    --type luks2 \
    --key-file "$ENC_KEY" \
    --cipher aes-xts-plain64 \
    --key-size 512

# Open the encrypted device
echo "[*] Opening LUKS volume as $MAPPER_NAME..."
sudo cryptsetup open "$NBD_DEVICE" "$MAPPER_NAME" --key-file "$ENC_KEY"

# Format the opened mapper as ext4
echo "[*] Formatting as ext4..."
sudo mkfs.ext4 "/dev/mapper/$MAPPER_NAME"

# Mount and copy files
echo "[*] Creating mount point $MOUNTPOINT..."
sudo mkdir -p "$MOUNTPOINT"

# Mount source drive
sudo qemu-nbd --connect=/dev/nbd2 $SOURCE_DISK
SOURCE_MOUNT_POINT="/mnt/clear_root"
sudo mkdir -p "$SOURCE_MOUNT_POINT"


echo "[*] Mounting..."
sudo mount "/dev/mapper/$MAPPER_NAME" "$MOUNTPOINT"
sudo mount /dev/nbd2 $SOURCE_MOUNT_POINT

echo "[*] Copying files from $SOURCE_DIR..."
sudo cp -a "$SOURCE_MOUNT_POINT/." "$MOUNTPOINT/"

echo "[*] Syncing..."
sync

# Unmount and clean up
echo "[*] Unmounting..."
sudo umount "$MOUNTPOINT"
sudo umount "$SOURCE_MOUNT_POINT"

echo "[*] Closing LUKS volume..."
sudo cryptsetup luksClose "$MAPPER_NAME"

echo "[*] Disconnecting NBD..."
sudo qemu-nbd --disconnect "$NBD_DEVICE"
sudo qemu-nbd --disconnect /dev/nbd2

echo "[*] Done."
