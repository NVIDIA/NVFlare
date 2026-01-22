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
Utility functions for AMPLIFY federated learning.
"""

import csv
import os
from datasets import load_dataset, Features, Value


def load_and_validate_csv(train_csv, test_csv, verbose=True, max_samples=None, seed=42):
    """
    Load and validate CSV files for training.
    
    Args:
        train_csv (str): Path to training CSV file
        test_csv (str): Path to test CSV file
        verbose (bool): Whether to print detailed information
        max_samples (int, optional): Maximum number of samples to use from train/test sets. 
                                     If None, uses all samples. Useful for quick testing.
        seed (int): Random seed for sampling (default: 42)
    
    Returns:
        dataset: HuggingFace dataset with 'train' and 'test' splits
    
    Raises:
        FileNotFoundError: If CSV files don't exist
        Exception: If there are issues loading the data
    """
    # Check if files exist
    if not os.path.exists(train_csv):
        # Provide helpful error message with guidance
        error_msg = f"Training data file not found: {train_csv}\n\n"
        
        # Check if this is for all-tasks scenario (looking for clientN_train_data.csv)
        if "client" in os.path.basename(train_csv) and "_train_data.csv" in train_csv:
            error_msg += "It looks like you're trying to run the all-tasks scenario.\n"
            error_msg += "You need to prepare the data with client splits first.\n\n"
            error_msg += "Run the following command from the examples/advanced/amplify directory:\n\n"
            task_name = os.path.basename(os.path.dirname(train_csv))
            error_msg += f"  python src/combine_data.py --input_dir ./FLAb/data/{task_name} \\\n"
            error_msg += f"      --output_dir ./FLAb/data_fl/{task_name} --num_clients 6 --alpha 1.0\n\n"
            error_msg += "Or prepare all tasks at once:\n\n"
            error_msg += "  for task in aggregation binding expression immunogenicity polyreactivity thermostability; do\n"
            error_msg += "      python src/combine_data.py --input_dir ./FLAb/data/$task \\\n"
            error_msg += "          --output_dir ./FLAb/data_fl/$task --num_clients 6 --alpha 1.0\n"
            error_msg += "  done\n"
        
        raise FileNotFoundError(error_msg)
    
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test data file not found: {test_csv}")
    
    if verbose:
        print(f"Loading dataset from: train={train_csv}, test={test_csv}")
        print(f"Data files verified. Train size: {os.path.getsize(train_csv)} bytes, Test size: {os.path.getsize(test_csv)} bytes")
    
    # Check CSV structure
    if verbose:
        print(f"Checking CSV file structure...")
        with open(train_csv, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            print(f"  CSV headers: {headers}")
            first_row = next(reader, None)
            if first_row:
                combined_sample = first_row.get('combined', 'N/A')
                if combined_sample and len(combined_sample) > 50:
                    combined_sample = combined_sample[:50] + "..."
                print(f"  First row sample: combined={combined_sample}, fitness={first_row.get('fitness', 'N/A')}")
    
    # Load dataset (only the columns we need)
    # IMPORTANT: Load fitness as string to avoid PyArrow casting errors with Unicode characters
    data_files = {"train": train_csv, "test": test_csv}
    if verbose:
        print(f"Loading only 'combined' and 'fitness' columns (fitness loaded as string)...")
    
    try:
        # Explicitly specify that fitness should be loaded as a string
        # This prevents automatic type inference which fails on Unicode characters
        features = Features({
            'combined': Value('string'),
            'fitness': Value('string'),  # Load as string, we'll clean and cast later
        })
        dataset = load_dataset(
            "csv", 
            data_files=data_files, 
            usecols=['combined', 'fitness'],
            features=features
        )
        if verbose:
            print(f"Dataset loaded successfully. Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")
    except Exception as e:
        print(f"Error loading dataset: {type(e).__name__}: {e}")
        if verbose:
            import traceback
            print(f"Full traceback:")
            traceback.print_exc()
        print(f"Please check that the CSV files exist and are properly formatted")
        raise
    
    # Check for and filter out invalid values, and convert fitness to float
    for split in ["train", "test"]:
        if dataset[split].features:
            original_size = len(dataset[split])
            
            # Define a validation and conversion function
            def is_valid_row(x):
                """Check if row has valid combined and fitness values."""
                combined = x.get('combined')
                fitness = x.get('fitness')
                
                # Check combined field
                if combined is None or combined == '':
                    return False
                
                # Check fitness field - can be None, empty, or non-numeric string
                if fitness is None or fitness == '':
                    return False
                
                # Try to convert fitness to float (handles string type)
                # Clean the string by removing all non-printable characters
                try:
                    # Convert to string and strip whitespace
                    fitness_str = str(fitness).strip()
                    # Remove zero-width spaces and other invisible Unicode characters
                    import unicodedata
                    fitness_clean = ''.join(c for c in fitness_str if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')
                    # Try to parse as float
                    float(fitness_clean)
                    return True
                except (ValueError, AttributeError):
                    return False
            
            # Filter invalid rows
            dataset[split] = dataset[split].filter(is_valid_row)
            
            # Clean fitness values before casting to remove invisible Unicode characters
            def clean_fitness(x):
                """Clean fitness string by removing invisible Unicode characters."""
                import unicodedata
                fitness_str = str(x['fitness']).strip()
                # Remove control characters (category C) except common whitespace
                fitness_clean = ''.join(c for c in fitness_str if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')
                x['fitness'] = fitness_clean
                return x
            
            dataset[split] = dataset[split].map(clean_fitness)
            
            # Convert fitness column to float type using cast
            new_features = dataset[split].features.copy()
            new_features['fitness'] = Value('float32')
            dataset[split] = dataset[split].cast(new_features)
            
            filtered_count = original_size - len(dataset[split])
            if filtered_count > 0 and verbose:
                print(f"  WARNING: {split} had {filtered_count} invalid rows (None/empty/non-numeric fitness values)")
                print(f"  Filtered and converted dataset. Remaining samples: {len(dataset[split])}")
    
    # Optionally sample a random subset for faster testing
    if max_samples is not None:
        for split in ["train", "test"]:
            if len(dataset[split]) > max_samples:
                if verbose:
                    print(f"  Sampling {max_samples} random samples from {split} (originally {len(dataset[split])} samples)")
                dataset[split] = dataset[split].shuffle(seed=seed).select(range(max_samples))
            elif verbose:
                print(f"  Keeping all {len(dataset[split])} samples from {split} (max_samples={max_samples})")
    
    return dataset
