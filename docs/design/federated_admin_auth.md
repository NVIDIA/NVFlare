# Federated Admin Authentication

## Goal

FLARE admin users can authenticate through a standard OIDC provider instead of
using an admin private key on the admin machine. The server validates the signed
OIDC ID token, maps provider roles or groups to one FLARE role, and creates a
normal FLARE admin session.

FL clients do not validate OIDC tokens. They trust the server-side authorization
decision by verifying a server-signed job authorization manifest before executing
directly deployed jobs.

## Minimum Supported Contract

- OIDC applies to admin users only.
- Site, server, and FL client PKI stays unchanged.
- Admin auth mode is binary: `cert` or `oidc`.
- OIDC mode disables certificate admin login for that deployment.
- OIDC admins are provisioned in the identity provider, not as FLARE project
  participants.
- FLARE trusts one configured OIDC issuer and one configured OIDC client ID.
- Roles and groups must be present in the signed ID token.
- Interactive login uses public-client Authorization Code + PKCE.
- Automation can provide an ID token through `id_token_env`.
- Server-signed job manifests are required for directly deployed jobs.

The implementation intentionally does not support device-code login,
confidential OIDC clients, server-side `userinfo`, configurable JWT algorithms,
or a separate resource-audience override. Those config keys fail fast rather
than being ignored.

## Runtime Flow

1. The admin CLI obtains an ID token through browser SSO or `id_token_env`.
2. The admin CLI sends only the ID token to the FLARE server over TLS.
3. The server validates issuer, signature, lifetime, required claims, and client
   binding.
4. The server extracts configured role and group claim paths from the ID token.
5. The server maps those roles and groups to one effective FLARE role.
6. The server creates a FLARE admin session from the normalized principal.
7. Existing authorization checks run against the effective FLARE role.
8. On job submission, the server signs a job authorization manifest.
9. FL clients verify that manifest before executing the received app.

## Main Components

- `nvflare.fuel.sec.principal.Principal`
  Normalized identity used by session and authorization code.
- `nvflare.fuel.sec.oidc.OidcAdminAuthProvider`
  Validates OIDC ID tokens and builds a `Principal`.
- `nvflare.fuel.sec.oidc.OidcTokenSource`
  Gets the admin ID token from config, environment, or browser SSO.
- `nvflare.fuel.sec.oidc.RoleMapper`
  Maps provider roles and groups to one FLARE role.
- `nvflare.fuel.sec.job_trust`
  Issues and verifies server-signed job authorization manifests.
- `nvflare.fuel.hci.server.login.LoginModule`
  Selects `cert` or `oidc` admin login mode and creates sessions.
- `nvflare.fuel.hci.client.credentials.AdminCredentials`
  Client-side credential strategy (`CertAdminCredentials` / `OidcAdminCredentials`)
  chosen once from config; a new admin auth method is one new strategy class.

In provisioning, the shared OIDC admin kit is a regular `Participant` marked
`cert_less` (created at `Project` construction), so all builders iterate
`project.get_admins()` uniformly and `CertBuilder` simply skips cert generation
for cert-less participants.

## Token Validation

The server validates ID tokens locally using the provider JWKS endpoint.
OIDC dependencies are optional and isolated to `nvflare.fuel.sec.oidc`.

Required checks:

- `iss` matches the configured issuer.
- `aud` includes the configured FLARE OIDC client ID.
- If `aud` has multiple values, `azp` must equal the configured client ID.
- Signature validates against a JWKS signing key.
- Signing algorithm is one of `RS256` or `ES256`.
- `iss`, `sub`, `aud`, `exp`, and `iat` are present.
- Token lifetime checks pass.
- At least one provider role or group maps to a valid FLARE role.

The authorization-code login flow sends a `nonce` and the admin CLI verifies it
in the returned ID token, protecting the client against token injection. The
server does not track `jti` or `nonce`: a captured ID token can be replayed to
create new admin sessions until its `exp`. This exposure is accepted by design
(the client-side token cache deliberately reuses the token until `exp`); use
short ID-token lifetimes at the IdP, and see the introspection TODO for
deployments that need stronger revocation.

JWKS lookup uses PyJWT `PyJWKClient`. Authorization-code login uses Authlib.

Optional dependency extra:

```toml
oidc =
    Authlib>=1.7.0
    PyJWT[crypto]>=2.8.0
    requests>=2.28.0
```

## Role Mapping

Role mapping is explicit by default. Exact FLARE role-name matching is available
only when `default_exact_name_mapping` is explicitly enabled.

Supported FLARE role targets:

```text
project_admin
org_admin
lead
member
```

If several provider roles map to FLARE roles, the effective role is selected by
fixed precedence:

```text
project_admin > org_admin > lead > member
```

Supported mapping shape:

```yaml
role_mapping:
  default_exact_name_mapping: false
  roles:
    provider_project_admin: project_admin
    provider_org_admin: org_admin
    provider_lead: lead
    provider_member: member
  groups:
    /flare/project-admins: project_admin
  precedence:
    - project_admin
    - org_admin
    - lead
    - member
```

## Configuration

Operators edit `project.yaml`. Provisioning splits the OIDC settings into
generated server and admin startup files.

Primary `project.yaml`:

```yaml
auth:
  admin:
    type: oidc
    admin_kit_name: admin
    oidc:
      issuer: https://idp.example.com/realms/nvflare
      client_id: nvflare-admin
      scope: openid profile email roles
      redirect_uri: http://127.0.0.1:8765/callback
      open_browser: true
      role_sources:
        - roles
        - realm_access.roles
        - resource_access.${client_id}.roles
      group_sources:
        - groups
        - member_of
      role_mapping:
        roles:
          provider_project_admin: project_admin
        groups:
          /flare/project-admins: project_admin
```

Generated `fed_server.json` keeps only server validation settings:

```json
{
  "auth": {
    "admin": {
      "type": "oidc",
      "oidc": {
        "issuer": "https://idp.example.com/realms/nvflare",
        "client_id": "nvflare-admin",
        "role_sources": ["roles", "realm_access.roles", "resource_access.${client_id}.roles"],
        "group_sources": ["groups", "member_of"],
        "role_mapping": {
          "roles": {
            "provider_project_admin": "project_admin"
          },
          "groups": {
            "/flare/project-admins": "project_admin"
          }
        }
      }
    }
  }
}
```

Generated `fed_admin.json` keeps only client login settings and omits
`client.key` and `client.crt`:

```json
{
  "format_version": 1,
  "admin": {
    "project_name": "nvflare",
    "server_identity": "server",
    "scheme": "grpc",
    "host": "server.example.com",
    "port": 8003,
    "connection_security": "tls",
    "auth_type": "oidc",
    "ca_cert": "rootCA.pem",
    "oidc": {
      "issuer": "https://idp.example.com/realms/nvflare",
      "client_id": "nvflare-admin",
      "scope": "openid profile email roles",
      "redirect_uri": "http://127.0.0.1:8765/callback",
      "open_browser": true
    }
  }
}
```

For non-interactive tests or automation, use only an ID-token environment
variable:

```json
{
  "id_token_env": "NVFLARE_OIDC_ID_TOKEN"
}
```

## Job Authorization Manifest

OIDC admin kits have no admin private key, so the server becomes the job
authorization authority for directly deployed jobs. After submit authorization,
the server writes and signs a job authorization manifest with existing server
identity material.

The manifest records:

- job id and app name
- canonical job content hash
- submitter principal
- effective FLARE role
- auth method and issuer
- token id or session id when available
- submit time, issue time, and expiry time
- study
- target sites
- server identity
- manifest schema version

Clients require the manifest on every directly deployed app (only hub jobs are
exempt) and reject a deploy whose manifest is missing, expired, signed by an
untrusted or out-of-validity server certificate, signed by the wrong server
identity, has an invalid signature, does not match the received app content, or
does not target the receiving site. This is not configurable: the server fails
a deploy with a clear error when it has no identity material to sign with,
rather than shipping a deploy every client would reject.

**Upgrade ordering.** Because a pre-OIDC server does not attach the manifest and
clients require it, the **server must be upgraded before the FL clients**.
Upgrading a client ahead of the server makes that client reject every deploy
("missing server job authorization") until the server is upgraded too. Upgrade
the server first, then roll out client upgrades.

Manifest timestamps are checked with a 5-minute clock-skew allowance and a
1-hour default TTL, so deployments require client clocks within roughly 5
minutes of the server clock; verification errors report the computed skew, and
the allowance is tunable per client via `job_authorization_clock_skew`
(seconds) in fed_client.json.

When `require_signed_jobs` is `true`, jobs submitted by OIDC admins are exempt
from the submitter-signature requirement because OIDC kits hold no signing key.
This is an explicit trust-model change: for such jobs the server itself is the
authorization authority, FL clients verify the server-signed manifest, and the
server executes the submitted (unsigned) server-side app under its policy
controls only (BYOC/privacy scopes, submit authorization). Operators who
require cryptographic submitter signatures on all code can set
`require_signed_jobs: "strict"` in fed_server.json, which rejects unsigned jobs
for everyone (OIDC admins included, restoring the pre-OIDC behavior), or keep
cert admin auth.

Jobs marked with `from_hub_site` remain outside this direct-deploy path and are
treated as a separate hub trust boundary.

## Security Model Notes

- **Transport.** OIDC admin consoles hold no client TLS certificate, so the
  admin listener must accept one-way TLS — but this never affects FL clients.
  OIDC admin auth requires a dedicated admin port (provisioning fails when
  `admin_port` equals the FL port); the server writes
  `admin_connection_security: tls` into fed_server.json, which applies a
  per-listener override to the admin listener only. The FL listener keeps the
  configured `connection_security` (mTLS by default) with full peer-CN
  endpoint-identity binding. The relaxed admin listener is marked admin-only:
  the cell layer rejects any non-admin endpoint that connects through it, so it
  cannot be used to bypass the FL listener's mTLS requirements.
- **Cert-less admin registration.** With OIDC enabled, an admin cell may
  register without a client cert (only via the TLS admin listener). The issued
  registration token is a cellnet transport credential only; every admin command
  additionally requires a session created by a validated OIDC login. Cert-less
  registrations are logged at INFO for audit. Admin registration tokens are
  bound to the cell origin recorded at registration: subsequent messages with a
  spoofed origin fail cellnet auth validation, the same binding regular clients
  get (bindings age out 24h after last use).
- **Study membership.** Study membership is enforced for OIDC principals the
  same as for cert admins: an OIDC user must be enrolled in a study (for
  example via the `add_study_user` command, using the principal's display name)
  before logging into it.
- **Sessions.** Admin sessions are capped by `auth.admin.max_session_lifetime`
  (seconds, default 8h; settable in project.yml and validated at provision
  time) and, for OIDC, additionally by the ID token `exp`. The admin client
  caches the IdP refresh token (0600, alongside the id_token) and silently
  renews expired id_tokens via the OAuth2 refresh grant, so new sessions need
  no browser after the first SSO; providers that require it need
  `offline_access` in the configured scope. Any refresh failure drops the
  stored refresh token and falls back to interactive login. Cached tokens
  survive logout (every one-shot CLI command closes its session, so clearing on
  logout would force a browser SSO per command); the cache is cleared when the
  server rejects the token. All sessions are invalidated on server restart (per-process session
  epoch), which also prevents replay of logged-out session tokens.

## TODO

- Optional token introspection for deployments that require stronger revocation
  behavior than local JWT validation.
- Operator documentation for concrete OIDC provider setup examples.
