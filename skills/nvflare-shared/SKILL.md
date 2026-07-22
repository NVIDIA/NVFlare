---
name: nvflare-shared
description: Shared NVFLARE conversion references and templates used by the other NVFLARE agent skills (conversion workflow, validation ladder, dependency install, model exchange, metrics/artifact reporting, and the custom aggregator template). Not a user-triggered skill; loaded via references from the conversion skills.
license: Apache-2.0
version: "0.1.0" # NVSkills CI bootstrap: no behavior change.
metadata:
  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"
  min_flare_version: "2.8.0"
  blast_radius: read_only
  status: internal
  tags:
    - nvflare
    - federated-learning
    - shared-references
  languages:
    - python
  frameworks:
    - nvflare
  domain: ml
---

# NVFLARE Shared Skill References

Internal, non-triggered skill that holds guidance and templates shared by the
NVFLARE conversion skills so the same rules are authored once. It is installed
alongside every NVFLARE skill and referenced by relative path; it is not
selected or invoked on its own.

## Contents

- `references/conversion-workflow.md` — non-standard conversion, rerun,
  data-location, export, and authorization guidance.
- `references/validation-evidence.md` — the local validation ladder.
- `references/dependency-install.md` — dependency ordering and host-permission
  guidance.
- `references/pytorch-model-exchange.md` — PyTorch-family model/state-dict
  exchange details.
- `references/pytorch-family-recipe-selection.md` — PyTorch-family recipe
  discovery, algorithm guide, and catalog-based selection rules.
- `references/runtime-output-guidance.md` — runtime/export output locations.
- `references/metrics-and-artifact-reporting.md` — metric and artifact reporting.
- `assets/aggregator.py` — the custom weighted-aggregator template.

Consuming skills load these with relative paths such as
`../nvflare-shared/references/conversion-workflow.md` and adapt
`../nvflare-shared/assets/aggregator.py` rather than duplicating the guidance.
