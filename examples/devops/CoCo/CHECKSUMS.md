# Download checksum audit

Audited against fresh downloads and registry manifests on 2026-08-07.

| Component | Exact validated object | SHA-256 | Checksum source |
|---|---|---|---|
| containerd 2.2.2 | `containerd-2.2.2-linux-amd64.tar.gz` | `2c08c99cbde73b3388c6d5da68e0bcaebc70c9174f2b14d785695e4401b3ede0` | [containerd release `.sha256sum`](https://github.com/containerd/containerd/releases/download/v2.2.2/containerd-2.2.2-linux-amd64.tar.gz.sha256sum) |
| Helm 3.21.3 | `helm-v3.21.3-linux-amd64.tar.gz` | `15e041a93a590dce8100f39385cd98c84a765c9e36aeeb9e2dc6ff9e4769e2e0` | [Helm `.sha256sum`](https://get.helm.sh/helm-v3.21.3-linux-amd64.tar.gz.sha256sum) |
| Cosign 2.6.2 | `cosign-linux-amd64` executable | `d437b8f0d30f5dec169337607fcfa0238de1348503e175f1bb5b94330b1ee409` | [Cosign release checksum list](https://github.com/sigstore/cosign/releases/download/v2.6.2/cosign_checksums.txt) |
| CNI plugins 1.8.0 | `cni-plugins-linux-amd64-v1.8.0.tgz` | `ab3bda535f9d90766cccc90d3dddb5482003dd744d7f22bcf98186bf8eea8be6` | [CNI release `.sha256`](https://github.com/containernetworking/plugins/releases/download/v1.8.0/cni-plugins-linux-amd64-v1.8.0.tgz.sha256), downloaded by stage 10 |
| Calico 3.32.1 | [`manifests/calico.yaml`](https://raw.githubusercontent.com/projectcalico/calico/v3.32.1/manifests/calico.yaml) | `a1df919d9721cf667accdc3e72848911b0cb25cfab7d2478ad0c996302c95744` | Pinned from the exact upstream manifest; stage 10 verifies it before cluster-admin apply |
| NVIDIA NVAT 1.2.2 | [`nvat-local-repo-ubuntu2404-1-2-local_1.0-1_amd64.deb`](https://developer.download.nvidia.com/compute/nvat/1.2.2/local_installers/nvat-local-repo-ubuntu2404-1-2-local_1.0-1_amd64.deb), the `NVAT_REPO_URL` artifact | `31b0a1646f2bbc08ee599d10dbae106124ef2903f39e37095b96493913b37657` | Pinned known-good hash confirmed against a fresh NVIDIA HTTPS download |
| google/go-tdx-guest | GitHub source archive at commit `d0438ad179370160a3b98d9703b1559dcd1ed5ee` | `5c0a76ad4cc9f780d1dc55cf6f6bd7bccf25d3c8f7b74b05cc478b9001f7b51b` | Pinned known-good hash confirmed against a fresh GitHub archive download; stage 50 builds the quote collector and verifier from this exact source |
| snpguest 0.10.0 | `snpguest` x86-64 release executable | `70e700465e3523e67dd5104583dc36cd11eef630c6f04c5b9ccafd6ba2e76ca0` | Pinned against the [virtee/snpguest v0.10.0 release](https://github.com/virtee/snpguest/releases/download/v0.10.0/snpguest); Stage 40 verifies it in the NVFlare image build |
| Intel Trust Authority CLI 1.10.1 | `trustauthority-cli-v1.10.1.tar.gz` release archive (POSIX tar despite the suffix) | `d3875adbee96268471c82dd54f012b726fa8d6eefdd8f3243c0e7650fb55ff4e` | Pinned against the [Intel v1.10.1 release](https://github.com/intel/trustauthority-client-for-go/releases/download/v1.10.1/trustauthority-cli-v1.10.1.tar.gz); Stage 40 uses `tar -xf` and checks the runtime version |
| Distribution Registry 2.8.3 image | `docker.io/library/registry@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373` | OCI manifest digest | Resolved from Docker Hub; stage 30 rejects tag-only `REGISTRY_IMAGE` values |
| Trustee KBS staged image | `ghcr.io/confidential-containers/staged-images/kbs@sha256:c128232d271c3bc6cdfbd57b1e585b4aaa0c8de6dd987dbf2731786f60405e25` | OCI manifest digest | Resolved from GHCR for the previously validated staged-image commit; stage 30 rejects tag-only `TRUSTEE_IMAGE` values |

The containerd and Helm values validate compressed release archives. They are intentionally different from the hashes of `/usr/local/bin/containerd` and `/usr/local/bin/helm` after extraction.

Kubernetes Debian packages are authenticated by APT repository metadata and its installed signing key rather than the download-cache helper. OCI images above use immutable registry manifest digests. Other version-pinned Helm/OCI artifacts are verified by their registry transport and manifests.
