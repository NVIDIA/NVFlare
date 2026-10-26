# CoCo integration-test tooling

These helpers and fixtures support developer acceptance tests. They are outside
the deployment example's four-step workflow.

- `acceptance/`: private evidence collection, finite validation jobs, observer
  checks, and negative fixtures.
- `mixed-tdx-snp/`: provisioning fixtures used by the mixed-runtime regression
  tests. Tests replace endpoint, credential, and build paths before use.

Run the unit regressions under `tests/unit_test/lighter/cc_provision/impl/`.
Hardware acceptance additionally requires an approved test cluster and existing
Trustee services; these files do not install infrastructure.

For deployment instructions, use
[the CoCo example](../../../../examples/devops/coco/README.md).
