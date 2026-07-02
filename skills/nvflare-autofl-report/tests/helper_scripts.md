# Helper Script Checks

The report helper is covered by
`tests/unit_test/tool/autofl_skill_report_test.py`.

Run the focused checks from the repository root:

```bash
pytest tests/unit_test/tool/autofl_skill_report_test.py -q
```

The tests exercise stopped-state admission, pending-candidate refusal, explicit
interruption confirmation, retained versus observed result selection,
Markdown/JSON artifacts with optional plot availability, max/min objectives,
literature outcome synthesis, candidate lineage, malformed contract sections,
metric provenance, budget and test-metric warnings, guard/plotter parity,
optional agent context, and operation outside a Git repository.
