# Private data format

No study data is tracked by this repository. Put private manifests outside the repository or under the Git-ignored `private_data/` directory.

Exact reproduction expects 40 CSV files: two encoders × five folds × four files per fold. Every CSV must contain these columns:

| Column | Meaning |
| --- | --- |
| `patient_id` | Stable patient identifier; one row per patient |
| `institution` | Source institution |
| `site` | One of `CBTN_CQU`, `Harvard`, or `EBRAINS` |
| `recurrence_label` | Binary patient label, 0 or 1 |
| `tumor_grade` | Metadata retained in predictions |
| `encoder` | `UNI` or `Virchow2` |
| `num_slides` | Number of slides in the patient bag |
| `slide_ids` | Semicolon-separated slide identifiers in fixed order |
| `feature_paths` | Semicolon-separated feature files in the same order |

Feature files may be HDF5/H5, PyTorch, or NumPy arrays supported by the feature loader. Each instance must have 1024 values for UNI or 2560 values for Virchow2. The runner checks file presence without loading full arrays before training.

Within every fold, the three training-client manifests and global test manifest must be disjoint and together contain exactly the same 804 patients. Across five folds, every patient must occur in the test set exactly once. The code uses the provided splits without regenerating or rebalancing them.

`feature_path_prefix_map` supports moving feature roots without editing private manifests. For example:

```json
{
  "manifest_root": "private_data/manifests",
  "feature_path_prefix_map": {
    "features_as_stored/uni": "private_data/features/uni",
    "features_as_stored/virchow2": "private_data/features/virchow2"
  }
}
```

The longest matching prefix is replaced. Patient rows and semicolon-separated slide order are preserved.
