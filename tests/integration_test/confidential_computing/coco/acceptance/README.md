# CoCo developer acceptance helpers

These tools support regression and hardware acceptance work. Normal deployment
uses [the CoCo workflow](../../../../../examples/devops/coco/README.md).

The tools consume existing cluster, Trustee, registry, application-image, and
NVFlare administration inputs. They do not prepare or install infrastructure.

- `prepare.py`: generate private CPU-only TDX test topologies.
- `application/generate_job.py`: create a finite validation job.
- `verify_federation.py`: check membership, fresh validation, and job results.
- `negative_fixtures.py`: generate controlled manifest mutations.
- `evidence.py`: maintain a private operator evidence ledger.
- `observer_builder.py`, `trusted_channel_audit.py`, and
  `trusted_proof_audit.py`: collect bounded test observations.

Each command exposes its inputs through `--help`. Keep credentials, raw tokens,
quotes, and private evidence outside the source tree. A recorded test result or
log match does not replace attestation or prove confidential execution.

The corresponding unit tests are under
`tests/unit_test/lighter/cc_provision/impl/tdx_*test.py`.
