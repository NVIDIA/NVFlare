# Federated data catalog

A catalog job reads the datasets configured for its Study. The researcher can select site and dataset names. The site controls each source URI, file mount, table selection and credential through its dataset configuration.

1. The controller sends each site its requested dataset names and remaining time budget. The executor checks the job marker, loads the site’s Study configuration and applies its read limits.
2. For each selected dataset, the executor calls `run_catalog_discovery`. The reader imports metadata from the configured source and generates an ODCS document in memory on each request. A failed dataset is reported as unavailable and an exhausted budget as skipped.
3. The controller associates results with the authenticated site, checks the reply’s version, allowed dataset names and contract bounds, and publishes standalone ODCS JSON files under `catalog/contracts/`. It atomically replaces `catalog/index.json` after writing the contracts.
4. The researcher calls `get_catalog(env, output_path="catalog_results.json")`. The helper reads the index and files, returning a version 2 snapshot with site outcomes and entries containing `site`, `dataset` and `contract`. The optional output file saves that same dict. The offline viewer can also read earlier version 1 snapshots.

See the [site reader](../../../app_opt/fdn/catalog/README.md) for source options.
