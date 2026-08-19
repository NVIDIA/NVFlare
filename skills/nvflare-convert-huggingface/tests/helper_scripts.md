# Helper Script Coverage

The Hugging Face model resolver is covered by
`tests/unit_test/tool/agent_skill_checks/conversion_templates_test.py`.

The focused tests cover explicit local/Hub source selection, bare relative local
paths, exit-zero cache misses, and one commit-pinned cache-aware resolution only
when download is authorized.
