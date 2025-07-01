from pathlib import Path

from nvflare.apis.fl_api.utils.import_fn import auto_import

# Auto-import trainers in trainers
trainer_dir = Path(__file__).parent
auto_import(trainer_dir)
