# CoCo public inputs

`registry-ca.crt` is the only service-produced file CoCo IT needs. Authenticate
its SHA-256 with the service administrator before running
`40-configure-registry-trust.sh`.

`kata-platform.env` contains only public platform pins. Receive
`kata-deploy-3.29.0.tgz` from provisioning_node and place it alongside that file, then
run `35-repin-kata-deployment.sh public/kata-deploy-3.29.0.tgz` from the kit root.
The installer checks the chart hash and pins the deployment image digest.
No platform archive, detached signature or platform signing public key is needed.

CoCo does not generate or approve measurements. These local pins support
reproducibility; the independent service's attestation and release policies are
the enforcement boundary against the untrusted cluster operator. Workload image
signature verification and the unchanged Pod handoff requirements still apply.

Do not place registry publisher credentials, Trustee certificates or private
keys, KBS tokens, image keys, signature keys, policies, build contexts, or
plaintext images in this directory.
