# Distributed Provisioning Examples

Two example modes are provided for the current CLI design:

- `interactive_mode/` for the step-by-step Bash walkthrough
- `scripted_mode/` for the scripted JSON-output flow
- `site.template.yml` for the single-site template

Key idea: CSR creation supports `--project-file <site.yml>` so both interactive and
scripted flows can use the same single-site input file without extra arguments.
For scripted flows, pass `--out-format json` so stdout remains machine-readable.
