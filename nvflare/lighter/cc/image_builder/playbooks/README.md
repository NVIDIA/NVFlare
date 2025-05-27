# Building NVFlare CVM with Ansible

This guide describes how to build and provision an NVFlare Confidential Virtual Machine (CVM) using Ansible playbooks.

Here are some design principles,

1. The `vars.yml` contains all variables/parameters needed for the CC project. All NVFlare build/provisioning systems should use values from this file. This is one single place for an end-user to define the project and CVM environment. This file is just a big dictionary, more keys can be 
added.

2. A pre-built CVM image must be available and running for Ansible to perform provisioning tasks. In the future, we may be able to automate the building of this CVM.

3. After the playbook run, the CVM is sealed (ssh shutdown, user deleted) and can only run
in the CC environment without any user interaction.


## Directory Structure

```
nvflare/lighter/cc/image_builder/playbooks/
├── provision_cvm.yml    # Main playbook for CVM provisioning
├── vars.yml             # Configuration variables
└── inventory.ini        # Target machine inventory
```

## Configuration

### 1. Inventory Setup

Edit `inventory.ini` to specify where CVM is running:

```ini
[nvflare_cvm]
your_target_ip ansible_user=your_user ansible_password=your_password
```

Note: Password is just for testing. It will replaced with keys or encrypted password in the future.

### 2. Variables Configuration

Edit `vars.yml` to configure your CVM:

```yaml
project:
  nvflare_version: 2.6.0
  jobs_source: /tmp/jobs
  data_source: /tmp/data
  custom_code: /tmp/custom

cvm:
  venv_path: /home/nvflare/venv
  sudo_password: nvflare
  service_source: /tmp/services
  required_packages:
    - "cryptsetup:2.2"
    - "lvm2:2.03"
    - "parted:3.3"
    - "iptables:1.8"
    - "systemd:245"
    - "dmsetup:1.02"
    # Additional security packages
    - "apparmor:3.0"
    - "selinux-utils:3.1"
    - "auditd:3.0"
    - "fail2ban:0.11"
    - "rkhunter:1.4"
    # Monitoring packages
    - "sysstat:12.0"
    - "prometheus-node-exporter:1.0"
    # Backup tools
    - "rsync:3.1"
    - "duplicity:0.8"
```

## Building the CVM

1. **Prepare the CVM**

   ```
   # Make a copy of the base CVM
   cp nvflare_cvm.qcow2 test_cvm.qcow2 
   # Start the VM
   ./start_vm.sh
   ```

2. **Run the Playbook**

   ```bash
   ansible-playbook -i inventory.ini provision_cvm.yml
   ```

   The playbook will perform all the tasks list in the playbook
   - Install required system packages
   - Set up Python virtual environment
   - Install NVFlare
   - Copy jobs to CVM
   - Copy data/model to CVM
   - Configure security settings, setting up the firewall
   - Install all the services needed for nvflare to run (TBD)

3. **Seal the CVM

   Disable SSH and remove the user. Not implemented yet
   after this step, nothing in the CVM can be changed anymore. 

