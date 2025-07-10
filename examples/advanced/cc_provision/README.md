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
│   └── shared.py
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
root_drive_size: 8
secure_drive_size: 2
data_source: /tmp/data

nvflare_version: "git+https://github.com/YuanTingHsieh/NVFlare.git@enhance_cc_provision"
nvflare_package: application_code.zip


cc_issuers:
  - id: snp_authorizer
    token_expiration: 3600
    path: "nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer"
  - id: gpu_authorizer
    token_expiration: 3600
    path: "nvflare.app_opt.confidential_computing.gpu_authorizer.GPUAuthorizer"

cc_attestation:
  check_frequency: 1800
  failure_action: stop_job

cvm_image_name: nvflare_cvm
```
