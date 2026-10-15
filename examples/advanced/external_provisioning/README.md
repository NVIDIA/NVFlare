# Public-root-only provisioning

Use an enterprise root certificate without giving its private key to provisioning.
Place the public PEM certificate beside `project.yml` as `rootCA.pem`, configure
all participants as in [project.yml](project.yml), then run:

```sh
nvflare provision -p project.yml -w workspace
```

Set the public root on `CertBuilder`:

```yaml
builders:
  - path: nvflare.lighter.impl.cert.CertBuilder
    args:
      public_root_cert: rootCA.pem
```

The path is relative to `project.yml` (the current directory for Python API callers).
The file must contain exactly one currently valid, self-signed CA certificate with
keyCertSign usage and no private key. Provisioning distributes it as `rootCA.pem`
in each startup kit and generates ordinary configuration. It neither issues
certificates nor reads credential history from earlier kits.

Every server, client and relay must set `external_cert: true`; the server must also
set `external_job_ca: true`. Every admin must have an `admin_cert_provider`. These
are existing credential-source settings. Provisioning fails with the participant
name and required setting if any credential would still need local issuance.
There is no provisioning issuer setting or CA adapter in this feature. The sample's
Step CA configuration belongs to the existing admin SSO provider and runs on the
admin machine when logging in; provisioning does not contact it.

Deploy credentials from your PKI into each participant's `startup` directory:

| Participant | Externally installed files |
| --- | --- |
| Server | `server.crt`, `server.key`, `job_ca.crt`, `job_ca.key` |
| Client | `client.crt`, `client.key`; also `server.crt`, `server.key` when it has a listener |
| Relay | `client.crt`, `client.key`, `server.crt`, `server.key` |
| Admin | Its configured certificate provider obtains credentials |

See the [provisioning guide](../../../docs/programming_guide/provisioning_system.rst#public-root-only-provisioning)
for certificate profiles, supported CA constraints and legacy
compatibility. The server signs job certificates locally using the supplied job CA.

Deliver kits and install private keys using your authenticated deployment process.
Process-launched jobs share the participant account/filesystem; this feature does
not add OS isolation. CC/HE content signing and hierarchical edge provisioning
are unsupported in public-root-only mode.
