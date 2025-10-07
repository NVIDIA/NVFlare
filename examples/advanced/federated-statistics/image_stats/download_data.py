import kagglehub
import shutil
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")


print("Path to dataset files:", path)

# Move downloaded data to output path
OUTPUT_DATASET_PATH = "/tmp/nvflare/image_stats/data"
output_path = Path(OUTPUT_DATASET_PATH)
if output_path.exists():
    shutil.rmtree(output_path)  # Remove if exists

shutil.move(path, OUTPUT_DATASET_PATH)
print(f"Dataset moved to: {OUTPUT_DATASET_PATH}")
