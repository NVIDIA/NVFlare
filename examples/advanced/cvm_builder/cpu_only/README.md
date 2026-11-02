# Minimal CPU-only CVM federation

Start here for a bare-metal CVM deployment. Both the server and client use the
same approved AMD SEV-SNP CPU-only generic image and the same project Trustee.
Build `/srv/nvflare/images/application.tar` with the parent example's
`docker/docker_build.sh`, replace the credential and image placeholders, then
run:

```bash
nvflare provision -p project.yml -w ./workspace
```

The parent mixed TDX/SNP GPU example is the advanced topology. Production use
still requires approval of the exact generic bundle and hardware acceptance.
