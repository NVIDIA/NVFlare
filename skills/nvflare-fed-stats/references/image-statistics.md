# Image Dataset Federated Statistics

The image path shares the tabular path's recipe, privacy filters, automatic
execution, and validation ladder; this reference holds everything that
differs. Generate the client from `assets/image_stats_client.py`, not from
the tabular template.

## Detection

Route to this path when the per-site data is image files: image extensions
(PNG/JPEG/BMP/TIFF, DICOM `.dcm`, NIfTI `.nii`/`.nii.gz`), image folder
layouts, or a datalist JSON pointing at image files. Mixed tabular+image
requests are two runs; report that and run them separately.

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
  round-2 histogram pass do not accumulate into the `Global` row (a product
  timing quirk) — report per-site values and sum them yourself.
- mean/sum/stddev/var/quantile/min/max over image sets are not supported by
  this skill version: report them as unsupported for image data rather than
  improvising pixel-pooled implementations.

## Histogram Range And Bins

- The intensity range comes from the pixel dtype/bit depth, which is
  legitimate static evidence: `[0, 256]` for 8-bit, `[0, 65536]` for
  16-bit, `[0, 1]` for normalized floats. Confirm from one sample image's
  dtype and state the choice; a user/README declaration overrides.
- The bin-cap cleanser compares bins to the site's *image count* (not pixel
  count): 20 bins needs >200 images per site. Size the default bin count to
  the smallest site and state the choice, exactly as in the tabular path.

## Loaders And Dependencies

- PIL (Pillow) covers PNG/JPEG/BMP/TIFF and is the template default, with
  grayscale conversion before histogramming — state that policy in the
  report; per-channel statistics are not supported in this version.
- DICOM requires `pydicom` (or MONAI's ITK reader); NIfTI requires
  `nibabel`. Preflight the import for the format actually present before
  generating the job, exactly like the fastdigest rule: on failure, fail
  closed with the product error, never silently skip files.

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
