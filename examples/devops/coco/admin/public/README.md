# Required deployment inputs (not included)

Receive `trustee.crt` and `registry-ca.crt` from the secure-services
administrator and authenticate their fingerprints independently. Receive
`approved-workload-launch-profile.json` from the trusted platform authority
through the provisioning coordinator; authenticate it and pin its SHA-256
in the private `platform.env` before stages 30/40.

Do not place private keys, credentials or raw attestation evidence here. The
public package contains no issued certificates or approved deployment values.
