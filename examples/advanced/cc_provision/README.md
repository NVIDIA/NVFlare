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

# Content inside CC configuration

```
compute_env: onprem_cvm

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
