# Orientation Routing Reference

`nvflare-orient` is the lead skill for ambiguous NVFLARE requests. It should
turn project evidence and user intent into one narrow next action.

## Evidence Sources

- The user-named entry point and the source files it directly uses.
- User-provided job folders, logs, artifacts, or deployment context.
- `../../nvflare-shared/references/framework-routing.md` for PyTorch-family
  training-loop ownership.

## Routing Rules

- Hugging Face or supported TRL trainer owning training:
  `nvflare-convert-huggingface`.
- Lightning `Trainer` owning training: `nvflare-convert-lightning`.
- Manual PyTorch training loop:
  `nvflare-convert-pytorch`.
- Conflicting owners or a trainer factory/wrapper whose owner is unclear: ask
  which entry point to federate before selecting a converter.
- Statistics, data summaries, histograms, or quantiles across sites, or an
  inspect result with `target_type` `tabular_dataset`/`image_dataset`
  (data-only targets recommend `nvflare-fed-stats` directly):
  `nvflare-fed-stats`.
- Generic "help me use FLARE here" with no clear workflow: read the relevant
  entry point, then
  recommend the narrowest skill.
- Existing FLARE job that fails or produces suspicious logs:
  `nvflare-diagnose-job`, not conversion.
- Existing FLARE job or project to optimize or improve — low accuracy, an
  underperforming metric, or hyperparameter/algorithm exploration:
  `nvflare-autofl`, not diagnosis.
- POC startup, production submission, Kubernetes deployment, or identity setup:
  route to the corresponding operations or deployment skill when available.
- Non-FLARE Python, web, data science, or generic ML questions: no FLARE skill.

## Output Shape

Summaries should name:

- target path inspected;
- strongest evidence found;
- recommended next skill or no-skill decision;
- unresolved semantic prerequisites and the validation expected in the next
  workflow.

Do not turn routing into implementation. Once the next skill is clear, hand off
instead of continuing with broad advice.
