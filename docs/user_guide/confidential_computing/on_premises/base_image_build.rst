.. _base_image_build:

##################################
CVM Base Image and Binary Building
##################################

This document explains how to build the prerequisite base images and binaries required by the CVM builder before running ``nvflare provision``.

The following artifacts must be built and placed in the image builder directory:

- ``base_images/ubuntu_base.qcow2`` — Ubuntu base disk image
- ``base_images/OVMF.amdsev.fd`` — Firmware with ``kernel-hashes=on`` support
- ``binaries/snpguest`` — Tool for interacting with the TEE
- ``binaries/kbs-client`` — Tool for communicating with Trustee KBS

.. note::

   In examples below, ``<builder_root>`` refers to the directory containing the ``cvm_build.sh`` script.

Build Ubuntu Base Image
=======================

The Ubuntu base image must be built on a **Ubuntu 25.04 host** with an **Ubuntu 24.04 guest**.

.. note::

   The OS versions differ between host and guest. This is the only tested combination.

The following instructions are adapted from NVIDIA's **Deployment Guide for SecureAI**:
https://docs.nvidia.com/cc-deployment-guide-snp.pdf

Download GPU Admin Tools
------------------------

.. code-block:: bash

   cd /shared/
   git clone https://github.com/NVIDIA/gpu-admin-tools

Autoload VFIO
-------------

Create ``/etc/modules-load.d/vfio.conf`` with the following content:

.. code-block:: text

   vfio
   vfio_pci

Restart the host to load the VFIO modules:

.. code-block:: bash

   sudo reboot

Download Ubuntu Installation Image
-----------------------------------

Download the ISO file for Ubuntu 24.04.2:

.. code-block:: bash

   cd /shared
   wget https://releases.ubuntu.com/24.04.2/ubuntu-24.04.2-live-server-amd64.iso

Create a Drive Image
--------------------

Create a drive image large enough to hold the OS. A minimum of 30GB is required to install Ubuntu and the GPU drivers. The builder will extend it as needed.

.. code-block:: bash

   qemu-img create -f qcow2 /shared/ubuntu_base.qcow2 30G

Install Ubuntu Guest
--------------------

Create the file ``/shared/launch_vm.sh`` with the following content:

.. code-block:: bash

   #!/bin/bash

   CORES=16
   MEM=32
   VDD_IMAGE=/shared/ubuntu_base.qcow2
   FWDPORT=9899
   CDROM=/shared/ubuntu-24.04.2-live-server-amd64.iso

   doecho=false
   docc=true
   sev=""

   while getopts "expc:" flag
   do
     case ${flag} in
       e) doecho=true;;
       x) docc=false;;
       p) FWDPORT=${OPTARG};;
       c) sev=${OPTARG};;
     esac
   done

   NVIDIA_GPU=$(lspci -d 10de: | awk '/NVIDIA/{print $1}')
   NVIDIA_PASSTHROUGH=$(lspci -n -s $NVIDIA_GPU | awk -F: '{print $4}' | awk '{print $1}')

   if [ "$doecho" = true ]; then
     echo 10de $NVIDIA_PASSTHROUGH > /sys/bus/pci/drivers/vfio-pci/new_id
   fi

   get_cbitpos() {
       modprobe cpuid
       EBX=$(dd if=/dev/cpu/0/cpuid ibs=16 count=32 skip=134217728 | tail -c 16 | od -An -t u4 -j 4 -N 4 | sed -re 's|^ *||')
       CBITPOS=$((EBX & 0x3f))
   }

   if [ "$docc" = true ]; then
     if [ -n "$sev" ]; then
          case "$sev" in
            sev|sev-es|sev-snp)
              SEV_MODE="$sev"
              USE_CC=true
              get_cbitpos
              ;;
            *)
              echo "Error: unsupported SEV mode '$sev'."
              echo "Use '-c' with valid options: sev, sev-es, sev-snp."
              echo "Or use '-x' to boot without CC modes"
              exit 1
              ;;
          esac
        fi
   fi

   qemu-system-x86_64 \
     -bios /usr/share/ovmf/OVMF.fd \
     -nographic \
     ${USE_CC:+  -machine confidential-guest-support=sev0,vmport=off} \
     ${USE_CC:+$( [ "$SEV_MODE" = sev ] && \
      echo "  -object sev-guest,id=sev0,cbitpos=${CBITPOS},reduced-phys-bits=1,policy=0x1" )} \
     ${USE_CC:+$( [ "$SEV_MODE" = sev-es ] && \
      echo "  -object sev-guest,id=sev0,cbitpos=${CBITPOS},reduced-phys-bits=1,policy=0x5" )} \
     ${USE_CC:+$( [ "$SEV_MODE" = sev-snp ] && \
      echo "  -object sev-snp-guest,id=sev0,cbitpos=${CBITPOS},reduced-phys-bits=1,policy=0x30000" )} \
     -vga none \
     -enable-kvm -no-reboot \
     -cpu EPYC-v4 \
     -machine q35 -smp $CORES -m ${MEM}G,slots=2,maxmem=512G \
     -drive file=$VDD_IMAGE,if=none,id=disk0,format=qcow2 \
     -device virtio-scsi-pci,id=scsi0,disable-legacy=on,iommu_platform=true,romfile= \
     -device scsi-hd,drive=disk0 \
     -netdev user,id=vmnic,hostfwd=tcp::$FWDPORT-:22 \
     -cdrom $CDROM \
     -device virtio-net-pci,disable-legacy=on,iommu_platform=true,netdev=vmnic,romfile= \
     -object iommufd,id=iommufd0 \
     -device pcie-root-port,id=pci.1,bus=pcie.0 \
     -device vfio-pci,host=${NVIDIA_GPU},bus=pci.1,iommufd=iommufd0,romfile=

Launch the VM to start the Ubuntu installation:

.. code-block:: bash

   chmod +x /shared/launch_vm.sh
   sudo /shared/launch_vm.sh -ex

Install a minimal Ubuntu 24.04. All required software will be installed by the builder later. After the guest OS is installed, Ubuntu will prompt you to reboot — the VM will terminate and return you to the host.

Save Base Image
---------------

Copy the VM image to the ``base_images`` folder:

.. code-block:: bash

   cp /shared/ubuntu_base.qcow2 <builder_root>/base_images

Fetch Firmware
==============

Check if ``OVMF.amdsev.fd`` is already available in ``/usr/share/ovmf``. If not, install it from the Ubuntu proposed repository:

.. code-block:: bash

   echo 'deb http://archive.ubuntu.com/ubuntu plucky-proposed main restricted universe multiverse' | \
     sudo tee /etc/apt/sources.list.d/plucky-proposed.list

   sudo tee /etc/apt/preferences.d/99-plucky-proposed <<'EOF'
   Package: *
   Pin: release a=plucky-proposed
   Pin-Priority: 100
   EOF

   sudo apt update
   sudo apt install -t plucky-proposed ovmf

Copy the firmware to the ``base_images`` folder:

.. code-block:: bash

   cp /usr/share/ovmf/OVMF.amdsev.fd <builder_root>/base_images

Build snpguest
==============

The ``snpguest`` tool is needed to interact with the TEE (Trusted Execution Environment).

.. code-block:: bash

   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source "$HOME/.cargo/env"

   sudo apt install -y build-essential

   # Checkout source code
   git clone https://github.com/virtee/snpguest.git
   cd snpguest
   git checkout v0.9.2
   cargo build -r

   cp target/release/snpguest <builder_root>/binaries

Build kbs-client
================

The ``kbs-client`` tool is used to interact with Trustee. Its version must match the Trustee server version. The tested commit is ``a2570329cc33daf9ca16370a1948b5379bb17fbe``.

.. code-block:: bash

   # Install dependencies
   sudo apt install -y pkg-config libtss2-dev

   # Checkout source code
   git clone https://github.com/confidential-containers/trustee.git
   cd trustee/tools/kbs-client
   git checkout a2570329cc33daf9ca16370a1948b5379bb17fbe

   # Build
   make -C ../../kbs cli

   cp ../../target/release/kbs-client <builder_root>/binaries
