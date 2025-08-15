# How to use CC provision

In project.yml, under each site add "cc_config: [file]", for example:

```yaml
participants:
  - name: site-1
    type: client
    org: nvidia
    cc_config: cc_site-1.yml
```

Then in the end of builders add:

```
builders:
  - path: nvflare.lighter.cc_provision.impl.cc.CCBuilder
```

Then use the following command to generate startup kits:

```bash
nvflare provision -p project.yml
```

# NVFlare application code package

For CC jobs, we don't allow custom codes, so we must pre-install those codes inside each CVM.
We utilize our nvflare pre-install command to do that.
 
First, we need to prepare the application_code_zip folder structure:

```bash
application_code_folder
├── application/                    # optional
│   └── <job_name>/
│               ├── meta.json       # job metadata
│               ├── app_<site>/     # Site custom code
│                  └── custom/      # Site custom code
├── application-share/              # Shared resources
|   └── simple_network.py           # Shared model definition 
└── requirements.txt       # Python dependencies (optional)
```

We have already prepared application-share folder and requirements.txt in this example.
We run the following command to create a zip folder so we can use that to build the CVM:

```bash
python -m zipfile -c application_code.zip application_code/*
```

# Content inside CC configuration

```
compute_env: onprem_cvm
cc_cpu_mechanism: amd_sev_snp
role: server

# All drive sizes are in GB
root_drive_size: 15
secure_drive_size: 2
data_source: /tmp/data

# Can be any pip-installable version string (e.g., "2.6.0", "latest", Git URL, etc.)
nvflare_version: "2.6.0"

# NVFlare application code package to be pre-installed inside the CVM
nvflare_package: application_code.zip
allowed_ports:
  - 8002
trustee_host: trustee-azsnptpm.eastus.cloudapp.azure.com
trustee_port: 8999

cc_issuers:
  - id: snp_authorizer
    path: nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer
    token_expiration: 150 # in seconds, needs to be less than check_frequency
  - id: gpu_authorizer
    path: nvflare.app_opt.confidential_computing.gpu_authorizer.GPUAuthorizer
    token_expiration: 150 # in seconds, needs to be less than check_frequency

cc_attestation:
  check_frequency: 300 # in seconds

```
