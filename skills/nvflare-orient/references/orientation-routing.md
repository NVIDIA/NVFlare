# Orientation Routing Reference

`nvflare-orient` is the lead skill for ambiguous NVFLARE requests. It should
turn project evidence and user intent into one narrow next action.

## Evidence Sources

- `nvflare agent inspect <path> --format json` for framework, FLARE usage,
  conversion state, safety findings, and recommended skills.
- `nvflare agent doctor --format json` for local CLI command surface, installed
  skill bundle, and ML framework dependency readiness.
- User-provided target files, job folders, logs, or stated deployment context.

## Routing Rules

- Existing PyTorch training loop needing FLARE conversion:
  `nvflare-convert-pytorch`.
- Generic "help me use FLARE here" with no clear workflow: inspect first, then
  recommend the narrowest skill.
- Existing FLARE job that fails or produces suspicious logs:
  `nvflare-diagnose-job`, not conversion.
- POC startup, production submission, Kubernetes deployment, or identity setup:
  route to the corresponding operations or deployment skill when available.
- Non-FLARE Python, web, data science, or generic ML questions: no FLARE skill.

## Output Shape

Summaries should name:

- target path inspected;
- strongest evidence found;
- recommended next skill or no-skill decision;
- validation or approval boundary before any mutating follow-up.

Do not turn routing into implementation. Once the next skill is clear, hand off
instead of continuing with broad advice.
