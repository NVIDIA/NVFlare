# Acknowledgements and provenance

## Public autoresearch inspiration

This starter bundle intentionally adapts the **public operating model** from:
- `karpathy/autoresearch` — https://github.com/karpathy/autoresearch

Specifically adapted ideas:
- `program.md` as the main agent-facing control plane
- a small editable surface for the coding agent
- branch-per-run setup
- a TSV experiment ledger (`results.tsv`)
- redirected `run.log`
- keep / discard / revert experiment loop
- the description of `program.md` as a lightweight “skill”
- the autonomous “never stop until interrupted” experiment-loop instruction
- `analysis.ipynb` / `progress.png` style experiment-progress visualization

## What was and was not copied

### Adapted
The bundle **adapts** the public repo's workflow and instruction style.
The current `program.md` in this bundle is a new NVFlare-specific rewrite based on those public ideas.

### Not copied verbatim
- No training code was copied from `karpathy/autoresearch`.
- No `AGENTS.md` or `CLAUDE.md` files were copied from that repo; the public repo does not appear to ship those files.
- The FL code in this bundle is based on NVFlare examples, not on `autoresearch` training code.

## Camyla literature-loop inspiration

The “think harder with literature” guidance in `program.md` is inspired by the public Camyla project page and repository:
- Camyla — https://yifangao112.github.io/camyla-page/
- Camyla GitHub repository — https://github.com/yifangao112/Camyla
- Camyla core package — https://github.com/yifangao112/Camyla/tree/main/camyla

Specifically adapted ideas:
- literature search as an explicit recovery mode when autonomous research stalls;
- diverse query generation and multi-source paper triage before proposing methods;
- extracting concrete research challenges before proposing methods;
- duplicate/null filtering before scoring new proposals;
- scoring proposals before branching;
- quality-weighted branch exploration;
- reflective memory to avoid repeating failures.

Relevant Camyla source locations consulted for the instruction design:
- README pipeline overview — https://github.com/yifangao112/Camyla/blob/main/README.md
- example idea-generation and assessment config — https://github.com/yifangao112/Camyla/blob/main/config_example.yaml
- literature search implementation — https://github.com/yifangao112/Camyla/blob/main/camyla/infrastructure/literature/multi_source_search.py
- challenge extraction/consolidation prompts — https://github.com/yifangao112/Camyla/tree/main/skills/agents
- QWBE experiment utilities — https://github.com/yifangao112/Camyla/tree/main/camyla/treesearch

This bundle does not copy Camyla code or its medical-segmentation task setup.
It adapts the high-level research-process pattern to a bounded NVFlare Auto-FL loop, where every literature-derived idea must still fit `mutation_schema.yaml` and preserve the v0 FL contract.

## NVFlare code lineage

This bundle is built on top of public NVFlare examples and utilities, especially:
- CIFAR-10 simulation README — https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/pt/cifar10-sim
- `examples/hello-world/hello-pt`
- `examples/advanced/cifar10/pt/cifar10-sim/cifar10_custom_aggr`
- `examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedavg`
- `examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedprox`
- `examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedopt`
- `examples/advanced/cifar10/pt/cifar10-sim/cifar10_scaffold`
- `nvflare.recipe.utils.add_cross_site_evaluation`
- `ValidationJsonGenerator`

The baseline exposes FedAvg, FedProx, FedOpt-style, and explicit SCAFFOLD benchmark knobs. SCAFFOLD follows the NVFlare example pattern of passing control-variate deltas and global controls through `FLModel.meta`, while preserving DIFF model uploads and the existing model key schema.

## Licensing note

The public `karpathy/autoresearch` repository is MIT-licensed.
This bundle includes attribution for the reused public workflow ideas and clearly separates them from newly written NVFlare-specific code and instructions.
