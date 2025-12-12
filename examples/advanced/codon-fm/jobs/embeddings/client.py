# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
client side embeddings extraction script
"""

import argparse
import os
import sys
import warnings

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


# Add project paths
sys.path.append("..")

from src.data.metadata import MetadataFields

# Import Encodon modules
from src.inference.encodon import EncodonInference
from src.inference.task_types import TaskTypes

# Import NVFlare
import nvflare.client as flare

# Fix random seed
torch.manual_seed(42)
np.random.seed(42)

print("✅ Libraries imported successfully!")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


def main(args):
    flare.init()
    sys_info = flare.system_info()
    print(sys_info)
    client_name = flare.get_site_name()

    checkpoint_path = args.checkpoint

    # Ignore the input model in this case
    input_model = flare.receive()

    # Load Pretrained Encodon Model
    model_loaded = False
    if os.path.exists(checkpoint_path):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Create EncodonInference wrapper
            encodon_model = EncodonInference(
                model_path=checkpoint_path,
                task_type=TaskTypes.EMBEDDING_PREDICTION,
            )

            # Configure model
            encodon_model.configure_model()
            encodon_model.to(device)
            encodon_model.eval()

            print(f"✅ Model loaded from: {checkpoint_path} on client {client_name}")
            print(f"Device: {device}")
            print(f"Parameters: {sum(p.numel() for p in encodon_model.model.parameters()):,}")

            model_loaded = True
        except Exception as e:
            print(f"Failed to load {checkpoint_path}: {e}")

    if not model_loaded:
        print("❌ Could not load any model. Please check checkpoint paths.")

    # Load Data
    if args.data_type == "train":
        data_path = os.path.join(args.data_prefix, client_name, "train_data.csv")
    else:
        data_path = os.path.join(args.data_prefix, "test_data.csv")

    data = pl.read_csv(data_path, separator="\t")
    data = data.with_columns(
        [
            pl.struct(["utr5_size", "cds_size", "tx_sequence"])
            .map_elements(
                lambda row: row["tx_sequence"][row["utr5_size"] : row["utr5_size"] + row["cds_size"]],
                return_dtype=pl.Utf8,
            )
            .alias("cds_sequence"),
            pl.struct(["utr5_size", "tx_sequence"])
            .map_elements(lambda row: row["tx_sequence"][: row["utr5_size"]], return_dtype=pl.Utf8)
            .alias("utr5_sequence"),
            pl.struct(["utr5_size", "cds_size", "tx_sequence"])
            .map_elements(lambda row: row["tx_sequence"][row["utr5_size"] + row["cds_size"] :], return_dtype=pl.Utf8)
            .alias("utr3_sequence"),
        ]
    ).with_row_index("id")
    output_path = data_path[:-4] + ".processed.csv"
    data.write_csv(output_path)

    # Load RiboNN dataset
    data_loaded = False
    if os.path.exists(output_path):
        try:
            data = pl.read_csv(output_path)
            print(f"✅ Loaded {len(data)} sequences from: {output_path}")
            print(f"Shape: {data.shape}")
            print(f"Key columns: {[col for col in ['id', 'cds_sequence', 'mean_te', 'fold'] if col in data.columns]}")

            # Show basic statistics
            te_stats = data.select(
                [
                    pl.col("mean_te").mean().alias("mean"),
                    pl.col("mean_te").std().alias("std"),
                    pl.col("mean_te").min().alias("min"),
                    pl.col("mean_te").max().alias("max"),
                ]
            )
            print("\nTranslation Efficiency stats:")
            print(f"  Mean: {te_stats['mean'][0]:.4f}")
            print(f"  Range: [{te_stats['min'][0]:.4f}, {te_stats['max'][0]:.4f}]")

            data_loaded = True
        except Exception as e:
            print(f"Failed to load {output_path}: {e}")

    # Data Preprocessing
    batch_size = args.batch_size
    if data_loaded and model_loaded:
        print("=== DATA PREPROCESSING ===")
        data = data.to_pandas()
        sequences = data["cds_sequence"].tolist()
        targets = data["mean_te"].values

        print(f"Processing {len(sequences)} sequences")

        # Extract embeddings
        print("\nExtracting embeddings...")
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size)):
                batch_seqs = sequences[i : i + batch_size]

                # Prepare batch
                batch_items = []
                for j, seq in enumerate(batch_seqs):
                    seq = seq.upper().replace("U", "T")
                    tokens = encodon_model.tokenizer.tokenize(seq)
                    input_ids = encodon_model.tokenizer.convert_tokens_to_ids(tokens)

                    # Truncate if needed
                    if (
                        len(input_ids) > encodon_model.model.hparams.max_position_embeddings - 2
                    ):  # Leave room for CLS/SEP
                        input_ids = input_ids[: encodon_model.model.hparams.max_position_embeddings - 2]

                    # Add special tokens
                    input_ids = (
                        [encodon_model.tokenizer.cls_token_id] + input_ids + [encodon_model.tokenizer.sep_token_id]
                    )
                    attention_mask = [1] * len(input_ids)

                    batch_items.append(
                        {
                            MetadataFields.INPUT_IDS: input_ids,
                            MetadataFields.ATTENTION_MASK: attention_mask,
                        }
                    )

                # Pad batch
                max_len = encodon_model.model.hparams.max_position_embeddings

                padded_input_ids = []
                padded_attention_masks = []

                for item in batch_items:
                    input_ids = item[MetadataFields.INPUT_IDS]
                    attention_mask = item[MetadataFields.ATTENTION_MASK]

                    # Pad
                    pad_len = max_len - len(input_ids)
                    input_ids.extend([encodon_model.tokenizer.pad_token_id] * pad_len)
                    attention_mask.extend([0] * pad_len)

                    padded_input_ids.append(input_ids)
                    padded_attention_masks.append(attention_mask)

                # Create batch tensor
                batch = {
                    MetadataFields.INPUT_IDS: torch.tensor(padded_input_ids, dtype=torch.long).to(encodon_model.device),
                    MetadataFields.ATTENTION_MASK: torch.tensor(padded_attention_masks, dtype=torch.long).to(
                        encodon_model.device
                    ),
                }

                # Extract embeddings
                output = encodon_model.extract_embeddings(batch)
                all_embeddings.append(output.embeddings)

        # Combine embeddings
        embeddings = np.vstack(all_embeddings)
        print(f"\n✅ Extracted embeddings: {embeddings.shape}")
        # Save embeddings
        embeddings_path = data_path[:-4] + ".embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"✅ Saved embeddings to: {embeddings_path}")
        # Save targets
        targets_path = data_path[:-4] + ".targets.npy"
        np.save(targets_path, targets)
        print(f"✅ Saved targets to: {targets_path}")
        success = True
    else:
        print("❌ Skipping preprocessing - data or model not loaded")
        success = False

    # Return success or failure
    output_model = flare.FLModel(
        params={"SUCCESS": int(success)},
    )
    print(f"site: {client_name} finished.")
    flare.send(output_model)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--checkpoint", type=str, default="/data/checkpoints/NV-CodonFM-Encodon-80M-v1")
    args.add_argument("--data_prefix", type=str, default="/data/federated_data")
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--data_type", type=str, default="train")
    args = args.parse_args()

    main(args)
