# Ephemeral Admin Certificates with step-ca

## Goal

Allow an admin to authenticate with OIDC and receive a short-lived FLARE admin
certificate without adding OIDC handling to the FLARE server or clients.

```text
admin CLI -> step CLI -> step-ca -> OIDC provider
admin CLI <- short-lived admin certificate/key
admin CLI -> existing FLARE mTLS login and job signing
```

The built-in `step_ca` provider delegates OIDC discovery, browser login, token
validation, claim mapping, and certificate issuance to step-ca.

## Trust Model

step-ca signs admin certificates with an intermediate CA rooted in the FLARE
project root. Existing servers and clients validate the resulting chain with
`rootCA.pem`. The FLARE server cannot mint an admin certificate or job
signature unless it controls step-ca, its signing key, or an admin private key.

The issued leaf certificate must contain the fields FLARE already consumes:

- `commonName`: authenticated admin identity
- `organizationName`: FLARE organization
- `unstructuredName`: `project_admin`, `org_admin`, `lead`, or `member`

The admin private key is generated on the admin machine by `step ca
certificate` and is not sent to the OIDC provider or FLARE server.

## Runtime Behavior

An ephemeral admin startup kit contains `ephemeral_admin_cert` instead of
static `client.crt` and `client.key` files. The admin client:

1. Loads a valid cached credential or invokes its configured provider.
2. Validates the certificate chain, validity, identity fields, allowed role,
   and certificate/private-key match.
3. Uses the existing certificate challenge, mTLS, authorization, and job-signing
   paths.
4. Reacquires credentials when the certificate enters its renewal window.

The certificate-chain support is implemented separately. This feature adds no
OIDC token format, server login mode, or server-signed job manifest to FLARE.

## Provisioning

Static and ephemeral admins can coexist in one `project.yml`:

```yaml
participants:
  - name: static-admin@example.com
    type: admin
    org: example_org
    role: project_admin

  - name: sso-admin-kit
    type: admin
    ephemeral_admin_cert:
      provider: step_ca
      renewal_window: 60
      provider_config:
        ca_url: https://step-ca.example.com
        provisioner: nvflare-admin-oidc
        cert_ttl: 24h
        command_timeout: 300
```

The ephemeral participant omits `org` and `role`; both come from the issued
certificate. Its name identifies a generic startup kit, not a user. The kit
contains `rootCA.pem` and provider configuration but no static admin
certificate or private key. Server and site startup kits are unchanged.

## Provider and Cache

`ephemeral_admin_cert.provider` is a built-in provider name or a
`module:function` path. A provider receives its configuration and the project
root certificate, and returns local certificate/key paths. FLARE applies the
same validation to built-in and custom provider results.

Valid credentials are cached per OS user under
`~/.nvflare/ephemeral_admin_certs`. The cache entry is bound to the provider
configuration and project root. Files are private to the OS user, concurrent
CLI processes serialize acquisition, and immutable credential directories
prevent cert/key replacement races.

The cache is required because each `nvflare` command starts a new process;
without it every command would repeat browser login. Users must not share an OS
account because that also shares its cached credential. Deleting the cache
forces a fresh OIDC login.

## step-ca Requirements

Operators configure step-ca, not FLARE, with:

- an intermediate CA signed by the FLARE project root
- an OIDC provisioner and matching IdP loopback redirect URI
- the OIDC client credentials and scopes needed by the X.509 template
- an X.509 template that writes the required FLARE identity fields
- a short maximum/default certificate duration, normally 24 hours
- renewal disabled so extending access requires another OIDC login

The root CA private key returns to offline storage after signing the
intermediate. The intermediate key remains with step-ca and may be protected by
an HSM/KMS. Neither private key is distributed to FLARE servers, sites, or
admins.

### Organization and Role Mapping

The step-ca template must map an exact, allowlisted IdP role to one
`(organization, FLARE role)` pair. Organization and role must not be accepted as
independent user-controlled claims.

Example mappings for one project and organization:

```text
nvflare-demo-example-project_admin -> (example, project_admin)
nvflare-demo-example-org_admin     -> (example, org_admin)
nvflare-demo-example-lead          -> (example, lead)
nvflare-demo-example-member        -> (example, member)
```

When several mapped roles for the same organization are present, the template
selects the highest privilege in this order:

```text
project_admin > org_admin > lead > member
```

Mappings that produce more than one organization are ambiguous and must fail
closed. Separate provisioners/templates per organization are the simplest
deployment model. The FLARE server does not map or rewrite certificate
organization or role values.

## Lifetime and Clone Behavior

The built-in provider requests a 24-hour certificate by default. FLARE does not
perform revocation checks, so disabling a user prevents new issuance but does
not invalidate an existing certificate. The lifetime must cover expected queue
and deployment delays because clients verify that the signing certificate is
still valid when the job is deployed.

`clone_job` copies the original submitter signature without contacting the
admin client. A clone therefore becomes unusable after the original certificate
expires. FLARE reports this before cloning when it can inspect the stored
certificate. A future client-assisted clone could re-sign and resubmit the job.

## References

- [step-ca](https://smallstep.com/docs/step-ca/)
- [`step ca certificate`](https://smallstep.com/docs/step-cli/reference/ca/certificate/)
- [step-ca provisioners](https://smallstep.com/docs/step-ca/provisioners/)
