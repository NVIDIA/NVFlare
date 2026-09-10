# Public files exported by the service administrator

After provisioning, export only these files:

- `trustee.crt`: copy of `~/trustee-public.crt`; distribute to the workload
  owner (`admin`). It authenticates the Trustee HTTPS endpoint.
- `registry-ca.crt`: copy of `~/.coco-publisher/registry-ca.crt`; distribute to
  both `admin` and `coco`. It authenticates registry port 5000.

The registry publisher username/password go only to `admin` through a separate
authenticated confidential channel. Never export the Trustee TLS private key,
registry CA private key, registry server private key, or KBS admin token.
