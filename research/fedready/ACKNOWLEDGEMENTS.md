# Acknowledgements and Provenance

FedReady combines NVIDIA FLARE orchestration with Codex-generated data
contracts, client-local adapters, and generated training code. It retains a
client-local VLM for raw-input guardrails, visual alignment review, rendering,
and automatic orientation repair.

This contribution intentionally excludes alternative coding-agent backends,
deterministic fallbacks, acquisition-quality review, sample-quality filtering,
and associated assets. `prepare_ref.sh` selects the required visual references
from locally available prepared records and records their content digests.

The retinal experiments use the 39-site retinal/fundus cohort derived from the
FedAgentBench dataset catalog (`arXiv:2509.23803v1`). The datasets are expected
to be available locally at each participating site.

FedReady and NVIDIA FLARE are licensed under Apache License 2.0. Dataset and
model licenses are not inherited by this code; operators must obtain and use
those artifacts under their own terms.
