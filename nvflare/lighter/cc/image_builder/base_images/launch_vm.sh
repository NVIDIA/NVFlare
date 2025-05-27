if [ $# -ne 2 ]; then
  echo "Usage: $0 <cvm_image> <applog_image>"
  exit 1
fi

VDD_IMAGE=$1
APPLOG_IMAGE=$2

UEFI_BIOS=/shared/OVMF.fd

#Hardware Settings
MEM=64 #in GBs
FWDPORT=9899

doecho=false
docc=true

while getopts "exp:" flag
do
        case ${flag} in
                e) doecho=true;;
                x) docc=false;;
                p) FWDPORT=${OPTARG};;
        esac
done

NVIDIA_GPU=$(lspci -d 10de: | awk '/NVIDIA/{print $1}')
NVIDIA_PASSTHROUGH=$(lspci -n -s $NVIDIA_GPU | awk -F: '{print $4}' | awk '{print $1}')

if [ "$doecho" = true ]; then
         echo 10de $NVIDIA_PASSTHROUGH > /sys/bus/pci/drivers/vfio-pci/new_id
fi

if [ "$docc" = true ]; then
        USE_HCC=true
fi

# SEV-ES:
#             -machine confidential-guest-support=sev0,vmport=off \
#             -object sev-guest,id=sev0,cbitpos=51,reduced-phys-bits=1,policy=0x5 \
#
# SEV-SNP:
#             -machine confidential-guest-support=sev0,vmport=off \
#             -object sev-snp-guest,id=sev0,cbitpos=51,reduced-phys-bits=1,policy=0x30000 \

#strace -f -o rob-sev-working-strace.txt qemu-system-x86_64 \

qemu-system-x86_64 \
             -machine confidential-guest-support=sev0,vmport=off \
             -object sev-snp-guest,id=sev0,cbitpos=51,reduced-phys-bits=1,policy=0x30000,kernel-hashes=off \
                -vga none \
                -monitor telnet:127.0.0.1:5555,server,nowait \
                -enable-kvm -nographic -no-reboot \
                -cpu EPYC-v4 \
                -machine q35 -smp 12,maxcpus=31 -m ${MEM}G,slots=2,maxmem=512G \
                -bios $UEFI_BIOS \
		-kernel /localhome/local-zhihongz/boot/vmlinuz \
                -initrd /localhome/local-zhihongz/boot/initrd.img \
                -drive file=$VDD_IMAGE,if=none,id=disk0,format=qcow2 \
                -drive file=$APPLOG_IMAGE,if=virtio,id=disk1,format=qcow2 \
		-netdev user,id=vmnic,hostfwd=tcp::2222-:22 \
                -device virtio-scsi-pci,id=scsi0,disable-legacy=on,iommu_platform=true,romfile= \
                -device scsi-hd,drive=disk0 \
                -netdev user,id=vmnic,hostfwd=tcp::$FWDPORT-:22 \
                -device virtio-net-pci,disable-legacy=on,iommu_platform=true,netdev=vmnic,romfile= \
                -device pcie-root-port,id=pci.1,bus=pcie.0 \
#                 -device vfio-pci,host=${NVIDIA_GPU},bus=pci.1,romfile= 
