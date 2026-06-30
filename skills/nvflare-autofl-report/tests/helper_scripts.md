# Helper Script Checks

The report helper is covered by
`tests/unit_test/tool/autofl_skill_report_test.py`.

Run the focused checks from the repository root:

```bash
pytest tests/unit_test/tool/autofl_skill_report_test.py -q
```

The tests exercise stopped-state admission, explicit interruption confirmation,
Markdown/JSON artifacts, max/min objectives, literature outcome synthesis,
candidate lineage, budget and test-metric warnings, optional agent context, and
operation outside a Git repository.

