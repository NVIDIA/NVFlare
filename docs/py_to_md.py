import os
import re
from pathlib import Path
import shutil

# --- CONFIG --- #
examples_dirs = ['../examples/hello_world']
filename_pattern = re.compile(r'example')  # same as sphinx_gallery_conf
ignore_pattern = re.compile(r'__init__\.py')
output_root = Path('tutorials_md')  # where to put .md files (relative to docs)

def convert_py_to_md(py_file, md_file):
    """Wrap code in markdown code fences"""
    with open(py_file, 'r') as f:
        content = f.read()
    with open(md_file, 'w') as f:
        f.write(f"# Tutorial: `{py_file.name}`\n\n")
        f.write("```python\n")
        f.write(content)
        f.write("\n```")

def generate_all_md():
    for examples_dir in examples_dirs:
        for py_path in Path(examples_dir).rglob("*.py"):
            if ignore_pattern.search(py_path.name):
                continue
            if not filename_pattern.search(py_path.name):
                continue

            rel_path = py_path.relative_to(examples_dir)
            md_path = output_root / rel_path.with_suffix('.md')
            md_path.parent.mkdir(parents=True, exist_ok=True)
            convert_py_to_md(py_path, md_path)

if __name__ == "__main__":
    if output_root.exists():
        shutil.rmtree(output_root)
    generate_all_md()
    print(f"✅ Markdown tutorials saved to: {output_root}")
