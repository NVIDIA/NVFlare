# Data-Derived Visual-Review References

Reference images are not distributed with this research contribution. Before
runtime preflight, select them from prepared local dataset records using a copy
of `meta/reference-sources.example.json`:

```bash
bash research/fedready/prepare_ref.sh /path/to/reference-sources.json
```

The source config selects one record for each of cup segmentation, disc
segmentation, disc detection, and binary glaucoma classification. Segmentation
and detection sources use JSONL manifests; classification may use a JSON list,
a `records` list, or a MONAI-style split dictionary. Records use the canonical
prepared-data fields:

- segmentation: `image` or `image_path`, plus `mask` or `mask_path`;
- detection: `image` or `image_path`, `boxes`, `labels`, and an explicitly
  declared `xyxy_abs` bounding-box format;
- classification: `image` or `image_path`, plus canonical integer `label` 0 or
  1.

The script copies the selected real image/target pairs into the canonical
final-form layout and writes a SHA-256-bound `manifest.json`. These references
calibrate visual review and are never used as federated training records.
Dataset licenses continue to apply to the locally prepared references.

To prepare the bundle elsewhere, pass the destination as the second argument
and set `FEDREADY_TASK_EXAMPLE_DIR` to that absolute directory before starting
FedReady processes.
