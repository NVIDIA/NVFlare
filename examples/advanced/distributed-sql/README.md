# Distributed SQL with NVFlare

A minimal `CollabRecipe` example: each client runs a fixed SQL query on its local
SQLite database; the server combines the summaries with SQL. All three sites
must succeed. Source rows stay local; no NVFlare framework changes are needed.

With your NVFlare environment active, run from this directory:

```bash
python prepare_data.py          # Three sample databases; refuses to overwrite
python job.py                  # Simulator
python job.py --env poc         # Local POC; services stop automatically
python -m pytest test_sql_query.py -q
```

Read `result.json` in the simulator's server job directory, or
`workspace/result.json` in the printed POC download directory.
The POC workspace is retained for inspection; its local service ports must be free.

The fixed query in [job.py](job.py) computes counts, sums, and averages by region.
Averages combine sums and non-NULL counts, so unequal site sizes and NULLs work.
The schema and sample data are in [prepare_data.py](prepare_data.py).
Both scripts accept `--db-root` (default `/tmp/nvflare/distributed-sql/data`);
each client opens `<db-root>/<site-name>.db` read-only. Edit `SITES`, `LOCAL_SQL`,
and `MERGE_SQL` to adapt the example. Export with:

```bash
python job.py --export --export-dir /tmp/sql-jobs
```

Databases are never included in the exported job. This worked example holds
summaries in memory, has no query row/deadline limits, and uses UNION ALL semantics
(overlapping records count twice). Summaries are visible to the server.
