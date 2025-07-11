import os
from pathlib import Path
import importlib


def auto_import(dir_path: Path):
    # Auto-import module for given module directory
    for file in os.listdir(dir_path.absolute()):
        if file.endswith(".py") and file not in ("__init__.py",):
            module_name = f"{__name__}.{file[:-3]}"
            importlib.import_module(module_name)
