# Public package boundary

This example contains reviewed scripts, public documentation, configuration
**templates**, policy source, tests and immutable public software pins. It
contains no live platform configuration, credentials, certificates, approved
measurements, generated Pod handoffs, private reports, VM images or recovery
archives. Historical operational records are not deployment inputs.

`example.com` endpoints and example accounts are illustrative. Configure real
endpoints and explicit target hostnames as described in [CONFIGURATION.md](CONFIGURATION.md).
Blank reference fields must come from trusted review, never from the adversarial
cluster. Template validation does not establish the trustworthiness of an input.

## Publish only a clean package

`PACKAGE-FILES.txt` is an explicit public-file allowlist. `validate-package.py`
rejects additional files, missing dependencies, stale checksums and known
lab-specific identifiers. `PACKAGE-SHA256SUMS` inventories the final public
package, not the private operational workspace. Authenticate its source
independently; hashes do not prove origin.

The `.gitignore` excludes commonly generated configuration, credentials,
certificates, release/evidence directories and archives. It is defense in depth,
not a secret scanner or authorization to publish ignored material. Review the
actual files before sharing. Never regenerate the allowlist from a used role
working directory. After an intentional source change, review the allowlist
and regenerate checksums only for those explicitly listed public files.

To export a validated clean package, use the allowlist rather than archiving
an entire working directory. From this folder, choose an output **outside** it:

```bash
tar --create --gzip --file=/path/to/coco-public.tar.gz \
  --verbatim-files-from --no-recursion --files-from=PACKAGE-FILES.txt \
  PACKAGE-SHA256SUMS
```

Replace the output path with your chosen location. The public inventory and
checksums travel with the archive; no generated files are implicitly included.

## Offline regression checks

Use a project virtual environment with Python 3.11+ and PyYAML. From a clean
package root, run its validator with bytecode output disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/venv/bin/python validate-package.py
```

For the complete handoff regression suite, put the stage-01 OPA 1.8.0 CLI on
`PATH` and point `TEST_OPA_BINARY` at that same checksum-pinned binary:

```bash
PYTHONDONTWRITEBYTECODE=1 TEST_OPA_BINARY=/usr/local/bin/opa \
  /path/to/venv/bin/python validate-package.py
```

Without those tools, two integration checks are explicitly skipped. The tests
exercise policy generation, reject modified handoff rules, evaluate resource
authorization with OPA, and mock privileged/network operations to test OPA
installation failures. They do not deploy services or perform attestation.

Generate secrets on their owning machines and transfer them only to the
necessary party. Admin receives its publisher credential, not KBS administration
or service TLS private keys. Secure services receives the workload decryption
key, not its signing private key. CoCo receives only public runtime/trust
inputs and the final Pod YAML. Public certificates become deployment-specific
inputs when generated; they are not committed here.

The lab-only teardown scripts and obsolete signed-bundle unpacker are excluded.
See [maintenance boundaries](trusted_system/TEARDOWN.md).
