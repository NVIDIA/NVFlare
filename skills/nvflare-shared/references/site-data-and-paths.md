# Site Data And Path Handling

Load this reference only when a conversion must generate site partitions or
resolve source data paths for generated clients.

## Site Data Partitioning

Preserve an existing user-provided site split. If no split exists, create
deterministic site-local training partitions unless the user explicitly asks all
sites to train on shared data. Prefer a seeded shuffle and use a stratified split
when classification labels are available. Do not default to stride or contiguous
splits because ordered data can produce biased sites.

For pandas `DataFrame` inputs, split positional row indices first, then build
each site frame with `df.iloc[positions]` or equivalent. Do not apply generic
array chunking directly to a `DataFrame`; library versions can return chunks
that no longer behave like data frames.

Make every array passed to an in-place shuffle writable. For label-stratified
partitioning, derive positional indices from the observed label column and copy
them before shuffling:

```python
positions = np.flatnonzero(frame[label_column].to_numpy() == label).copy()
rng.shuffle(positions)
site_frame = frame.iloc[positions]
```

Do not shuffle a possibly read-only array returned by `Index.to_numpy()`, and do
not pass positional indices to `DataFrame.loc`.

Report the split policy, seed, site count, and any reason stratification was not
used. Do not copy private site data into generated artifacts unless the user
explicitly requests it.

## Data Location

Pass the data location into the generated client as a configurable `train_args`
value, or through `per_site_config` when sites need different paths. Never
hardcode it inside `client.py`. Point at the original dataset rather than a copy
inside the NVFLARE run workspace.

Resolve a relative source data path in `job.py` against the original
source-project root before recipe construction. Pass the resolved value through
`train_args` or `per_site_config`; the client must consume it unchanged. Do not
reinterpret it relative to packaged `client.py`, the export directory, or the
simulator process working directory. Validate relative-path conversions from a
fresh caller working directory.

When `per_site_config` supplies `train_args`, each site value completely
replaces recipe-level `train_args`; it is not a fragment to merge. Compose the
shared options and site data path into every override, then apply the validation
in `pytorch-family-recipe-construction.md` before a full simulation.

An absolute path is acceptable only as the runtime-supplied value or the default
of a configurable argument. Do not bake a fixed absolute path into generated
client code. Report that real deployment requires every site to configure its
own local data location.
