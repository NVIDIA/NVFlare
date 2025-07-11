from pathlib import Path

from experimental.fl_api.common.utils.import_fn import auto_import

# Auto-import trainers in trainers
strategies_dir = Path(__file__).parent
auto_import(strategies_dir)
