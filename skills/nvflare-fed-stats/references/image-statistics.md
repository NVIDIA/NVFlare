# Image Dataset Federated Statistics

The image path shares the tabular path's recipe, privacy filters, automatic
execution, and validation ladder; this reference holds everything that
differs. Generate the client from `assets/image_stats_client.py`, not from
the tabular template.

## Detection

`nvflare agent inspect <path> --format json` classifies image data:
`target_type: image_dataset` with a `dataset` block carrying the extension
census, per-site file counts, and a sampled `pixel_depth` (e.g. `uint8`).
Route on that output rather than re-deriving it; `counts_approximate:
true` means the walk hit its file limit — verify site counts directly
before bin-cap decisions. Datalist-JSON layouts may classify as
`unknown_target`; read the datalist as declared layout evidence in that
case. Companion tabular metadata (a `labels.csv` beside the scans) keeps
the dataset `image` and appears per site as `tabular_companions`: mention
it in the report, never treat it as a statistics target — but offer a
follow-up tabular statistics run over the labels (label shift across
sites is a classic QA question). Genuinely mixed
data — an image-only site among tabular sites, or materially both
modalities — reports `modality: mixed` without routing (`target_type`
stays `unknown_target`): read the dataset block, report the split, and
run the two modalities as two separate jobs.

## Supported Statistics (Image)

The image feature is pixel `intensity`; the meaningful statistic set is:

```python
statistic_configs = {
    "count": {},          # number of image files discovered at the site
    "failure_count": {},  # unreadable/corrupt files; MUST be configured to appear
    "histogram": {"*": {"bins": 20, "range": [0, 256]}},
}
```

- `count` is the number of discovered image files, not pixels; unreadable
  files are only detected during the histogram pass, so `count` includes
  them and `failure_count` reports them.
- `failure_count` is per-site diagnostic only: failures detected during the
  round-2 histogram pass do not accumulate into the `Global` row (product
  defect, tracked as NVFlare issue #4876) — report per-site values and sum
  them yourself until it is fixed.
- mean/sum/stddev/var/quantile/min/max over image sets are not supported by
  this skill version: report them as unsupported for image data rather than
  improvising pixel-pooled implementations.

## Histogram Range And Bins

- The intensity range comes from the pixel dtype/bit depth, which is
  legitimate static evidence: `[0, 256]` for 8-bit, `[0, 65536]` for
  16-bit, `[0, 1]` for normalized floats. Use the dataset block's sampled
  `pixel_depth` and state the choice; a user/README declaration overrides.
- The bin-cap cleanser compares bins to the site's *image count* (not pixel
  count): bins must be under round(10% of effective count), so 20 bins
  first passes at 206 readable images per site. Size the default bin count
  using the dataset block's `site["image_files"]` — NOT `data_files`,
  which also counts companion label files — against the smallest site,
  and state the choice, exactly as in the tabular path.

## Loaders And Dependencies

- PIL (Pillow) covers PNG/JPEG/BMP/TIFF and is the template default, with
  grayscale conversion before histogramming — state that policy in the
  report; per-channel statistics are not supported in this version.
- DICOM requires `pydicom` (or MONAI's ITK reader); NIfTI requires
  `nibabel`. Preflight the import for the format actually present before
  generating the job, exactly like the fastdigest rule: on failure, fail
  closed with the product error, never silently skip files. Any format
  whose loader yields a pixel array is supported — the loader is the only
  gate.
- CT DICOM correctness: stored values become Hounsfield Units only after
  `RescaleSlope * value + RescaleIntercept`; apply the rescale in the
  loader and use a declared HU range (e.g. `[-1024, 3072]`) — never
  histogram raw stored values as calibrated intensities.
- Count semantics for medical formats: `count` is files, which for DICOM
  series means slices (not studies/patients) and for NIfTI means whole
  volumes; say which in the report. Cross-site intensity histograms also
  confound scanner/protocol calibration with population differences —
  include that caveat alongside the case-mix one.

## Validation Specifics

The ladder is unchanged; the statistics rungs adapt as:

- **Completeness** — the output hierarchy is
  `{"intensity": {statistic: {site: {dataset: value}}}}` with every site
  plus `Global` under each configured statistic.
- **Per-site parity** — recompute one site's histogram with an
  agent-authored snippet (same loader, same grayscale conversion, same
  bins/range) and compare bin counts exactly; count must equal the site's
  discovered-file count.
- **Global parity** — `Global` histogram bin counts equal the element-wise
  sum of the site bin counts; `Global` count equals the sum of site counts.
- **Failure reporting** — nonzero `failure_count` at a site is a data-quality
  finding to report (corrupt or unreadable files), not a run failure; the
  affected images are simply absent from that site's histogram.

Report shape follows the tabular path (aggregates only — bin counts and
file counts, never pixel data or file-name listings that could identify
patients), with the same case-mix caveat for `Global`.
