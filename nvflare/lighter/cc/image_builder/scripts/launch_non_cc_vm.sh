# Check exact number of arguments
if [ $# -ne 4 ]; then
  echo "Usage: $0 <root_drive> <applog_drive> <DATA_DRIVE> <log_file>"
  exit 1
fi

ROOT_DRIVE=$1
APPLOG_DRIVE=$2
DATA_DRIVE=$3
LOG_FILE=$4

sudo qemu-system-x86_64 \
    -bios OVMF_AMD.fd \
    -kernel bzImage \
    -initrd initramfs.cpio.gz \
    -append "console=ttyS0 root=/dev/vda1 rw net.ifnames=0 biosdevname=0 amd_iommu=on swiotlb=2097152,force" \
    -nographic \
    -machine q35,confidential-guest-support=sev0,vmport=off \
    -object sev-snp-guest,id=sev0,cbitpos=51,reduced-phys-bits=1,policy=0x30000,kernel-hashes=on \
    -vga none \
    -enable-kvm -no-reboot \
    -cpu EPYC-v4 \
    -machine q35 -smp cores=8,threads=2,sockets=2 -m 30G,slots=2,maxmem=512G \
    -serial file:$LOG_FILE \
    -drive file=$ROOT_DRIVE,if=none,id=disk0,format=qcow2 \
    -device virtio-blk-pci,drive=disk0,serial=rootfs \
    -drive file=$APPLOG_DRIVE,if=none,id=disk1,format=qcow2 \
    -device virtio-blk-pci,drive=disk1,serial=applog \
    -drive file=$DATA_DRIVE,if=none,id=disk2,format=qcow2 \
    -device virtio-blk-pci,drive=disk2,serial=user_data \
    -device virtio-scsi-pci,id=scsi0,disable-legacy=on,iommu_platform=true,romfile= \
    -netdev user,id=vmnic,hostfwd=tcp::2222-:22,hostfwd=tcp::8002-:8002,hostfwd=tcp::8003-:8003 \
    -device virtio-net-pci,disable-legacy=on,iommu_platform=true,netdev=vmnic,romfile=  \
    -device pcie-root-port,id=pci.1,bus=pcie.0 \
    -device vfio-pci,host=81:00.0,bus=pci.1,romfile=
