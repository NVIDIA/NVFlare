# PyTorch-Family Recipe Selection

This reference is shared by every PyTorch-family conversion skill (plain PyTorch
and Lightning). Lightning is a PyTorch-family training framework, not a separate
recipe family, so both select from the same catalog. The training framework
identifies the client-side conversion; it does not determine the federated
workflow. Choose the recipe from the user's FL intent and aggregation,
state-exchange, privacy, and site-role requirements.

## Discover Recipes

Run the local recipe catalog before creating or updating `job.py`:

```bash
nvflare recipe list --framework pytorch --format json
```

Use the returned recipe metadata as the source of truth for recipe names,
modules, classes, algorithms, aggregation mode, state exchange, privacy and
privacy-compatibility metadata, and optional dependencies.

After selecting a candidate recipe, inspect its parameters:

```bash
nvflare recipe show <recipe-name> --format json
```

Prefer recipe CLI metadata before Python introspection. If Python introspection
is still needed, do not guess top-level exports: the base `Recipe` class lives in
`nvflare.recipe.spec`, not `nvflare.recipe`.

## Quick Algorithm Guide

If the user does not know which FL algorithm they need, explain the choices in
plain language before editing `job.py`:

- FedAvg: the default starting point for most horizontal FL jobs. Each site
  trains locally, sends model weights or updates, and the server averages them.
  Use this when the user simply asks to federate normal training.
- FedAvg with HE: FedAvg plus homomorphic encryption support for protected
  aggregation. Use only when the user asks for HE or encrypted aggregation.
- FedProx: FedAvg-style training with a proximal term in the client loss to
  improve stability when site data or compute behavior is very different.
- FedOpt: server-side optimizer variants such as FedAdam, FedYogi, or FedAdagrad.
  Use when the user wants server optimizer control or better behavior than plain
  averaging on heterogeneous data.
- SCAFFOLD: adds control variates to reduce client drift on non-IID data. Use
  when the user specifically asks for SCAFFOLD or drift mitigation.
- Cyclic: sends the model through clients sequentially instead of aggregating
  updates on the server. Use when the requested workflow is client-to-client or
  cyclic weight transfer.
- Swarm Learning: peer/client-parent aggregation topology instead of a normal
  server-centered FedAvg topology. Use when the user asks for swarm learning.
- FedEval: evaluation-only. Use when the user wants to distribute a checkpoint
  to sites and collect metrics without federated training updates.

For deeper background, see the algorithm papers for
[FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html),
[FedProx](https://arxiv.org/abs/1812.06127),
[FedOpt](https://openreview.net/forum?id=LkFG3lB13U5),
[SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html), and
[Swarm Learning](https://www.nature.com/articles/s41586-021-03583-3). For
Cyclic recipes, use the local catalog and
`nvflare recipe show <selected-cyclic-recipe> --format json`.

## Selection Rules

- Use `fedavg-pt` for standard horizontal federated training where clients train
  the same model locally and the server aggregates model weights or weight diffs
  across rounds. This is the default fast path.
- Use `fedeval-pt` for evaluation-only jobs that send a checkpoint to sites and
  collect metrics without local training updates.
- For any other algorithm or privacy requirement, do not assume a recipe name:
  select the catalog recipe whose metadata matches the request, confirmed with
  `nvflare recipe list --framework pytorch --format json` and
  `nvflare recipe show <recipe> --format json`. Match on the catalog fields the
  CLI actually exposes: `algorithm`, `aggregation`, `state_exchange`, `privacy`,
  and `privacy_compatible`. Do not match on `topology`; use topology wording in
  the request as intent that maps to exposed catalog fields, recipe semantics,
  or required role parameters.
- Privacy is safety-critical: a homomorphic-encryption request must select a
  recipe whose `privacy` includes `homomorphic_encryption` (for example
  `fedavg-he-pt`), or a recipe whose `privacy_compatible` includes
  `homomorphic_encryption` only when the generated job also configures HE. Never
  map an HE request to a `privacy: []` recipe such as `fedavg-pt`; when no
  catalog recipe matches the requested privacy, ask or fail closed rather than
  dropping the encryption requirement.
- HE recipes cannot be validated locally by a conversion skill: `fedavg-he-pt`
  (`FedAvgRecipeWithHE`) rejects `SimEnv` because HE needs provisioned startup
  kits (`HEBuilder`) and a `PocEnv`/`ProdEnv`, and those environments are
  outside conversion scope per `conversion-workflow.md`. For an HE request, do
  not generate a `job.py` that calls `recipe.execute(SimEnv(...))`, and do not
  swap to a non-HE recipe to make simulation pass. Ask the user or fail closed:
  deliver the generated sources (and an exported job if export is in scope),
  report the job as unvalidated, and state that running it requires HE
  provisioning with `PocEnv`/`ProdEnv`.
- Current names are examples to verify against the catalog, not an authoritative
  mapping: `fedavg-he-pt` (FedAvg, `privacy: [homomorphic_encryption]`,
  `privacy_compatible: [homomorphic_encryption]`),
  `fedprox-pt` (FedProx / proximal loss), `fedopt-pt` (server-side optimizer
  variants such as FedAdam / FedYogi / FedAdagrad), `scaffold-pt` (SCAFFOLD
  control variates / client-drift mitigation), `cyclic-pt` (sequential
  client-to-client transfer), `swarm-pt` (swarm / peer aggregation topology).
- Ask the user before choosing when the requested FL workflow is not clear.

## Non-FedAvg Recipe Rules

The FedAvg fast path is not a universal job template. When the user asks for
FedOpt, FedProx, SCAFFOLD, Cyclic, Swarm Learning, FedEval, encryption, or a
topology-specific workflow:

- use `nvflare recipe show <recipe-name> --format json` for the selected recipe;
- supply parameters marked `"required": true`;
- leave optional parameters at defaults unless the user request, source code, or
  validation result requires them;
- keep generated source names consistent with the loading skill and runtime
  locations consistent with `runtime-output-guidance.md`;
- keep shared generated files on all clients unless the recipe semantics or user
  request require site-specific roles, scripts, arguments, or launch settings;
- ask before choosing when recipe intent or topology is ambiguous.

For recipes with topology roles, such as cyclic ordering, swarm roles, vertical
data ownership, or label-owner style workflows, do not collapse the topology into
standard FedAvg. Preserve the selected recipe's required role or site parameters
and report any missing user input before validation.

## Framework-Specific Job Construction

This reference covers recipe discovery, algorithm choice, and selection rules.
For the framework-specific `job.py` construction — the fast-path recipe code,
model config, portable imports, per-site deployment, and execution mode — follow
the loading skill's own recipe reference (the recipe-selection reference for
plain PyTorch; the Lightning conversion reference for Lightning). Export handling is
shared: follow `conversion-workflow.md` for `--export`, `--export-dir`, and
local command-line parser behavior.
