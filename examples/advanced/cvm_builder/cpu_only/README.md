# Minimal CPU-only CVM federation

Start here for a bare-metal CVM deployment. Both the server and client use the
same approved AMD SEV-SNP CPU-only generic image and the same project Trustee.
Build the CPU-specific application archive from this directory:

```bash
../docker/docker_build.sh Dockerfile.cpu nvflare-site-cpu /srv/nvflare/images/application.tar
```

Replace the credential and image placeholders before provisioning. If copying
this example outside the repository, change `cvm_builder_dir` in `cc_project.yml`
to the absolute installed builder path, for example
`/opt/NVFlare/nvflare/lighter/cc/image_builder`; the checked-in relative path
works only at its repository location. All relative project-service paths
resolve against `cc_project.yml`, not the shell's current directory.

`server.example.invalid` is deliberately a non-resolving example name. Replace
the `192.0.2.10` placeholder in both participant `hosts_entries` mappings with
the server CVM host's client-reachable IP (the host forwards port 8002 into the
CVM), or remove the mappings and configure trusted DNS. These authenticated
entries reach both the guest and application container; they do not change the
admin machine's resolver or the measured KBS hostname. Configure those separately.

Materialize the generic `cvm_image` as the provisioning user, so that user can
read `profile_set.json`; sudo for the later vault build does not grant access to
the earlier provisioning inspection. Follow the
[generic-image permissions guidance](../../../../nvflare/lighter/cc/image_builder/USER_GUIDE.md#generic-images-used-by-provisioning).
See [archive compression and sizing](../docker/README.md) before changing
`vault_size_gib`. Then run:

```bash
nvflare provision -p project.yml -w ./workspace
```

The parent mixed TDX/SNP GPU example is the advanced topology. Production use
still requires approval of the exact generic bundle and hardware acceptance.
