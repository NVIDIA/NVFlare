# Gen3 data access for an NVFlare study

This example trains a small federated linear regression model using synthetic CSV files downloaded from Gen3. The training script uses `Gen3Auth` and `Gen3File.download_single()` directly. Each client downloads its data once, then trains locally using the NVFlare Client API.

## Install and prepare data

Run these commands from this directory, with NVFlare installed in development mode:

```bash
python -m pip install -r requirements.txt
python prepare_data.py --output /tmp/gen3-demo-data
```

This creates two CSV files with 128 samples each and a manifest assigning object IDs and filenames to `site-1` and `site-2`.

## Configure Gen3 and client credentials

1. Upload the CSVs to protected storage and register their manifest object IDs in Gen3 Indexd. The registered `file_name` must match the manifest filename.
2. Create a Gen3 application principal for each site's study workload and grant access to its assigned objects. Enable the OAuth `client_credentials` grant and Fence's `CLIENT_CREDENTIALS_ON_DOWNLOAD_ENABLED` setting.
3. At each site, supply its endpoint, client ID, and client secret through study runtime settings:

```yaml
format_version: 2
studies:
  gen3-demo:
    env:
      GEN3_ENDPOINT: https://gen3-test.example.org
      GEN3_CLIENT_ID: replace-with-this-sites-application-client-id
    secret_env:
      GEN3_CLIENT_SECRET:
        source: gen3-demo-credentials
        key: client_secret
```

Create the `gen3-demo-credentials` Kubernetes Secret with key `client_secret` using your organization's secret tooling. Copy [study_runtime.yaml](study_runtime.yaml) into the site's `local/` directory, merging with existing settings if needed. Install the example's dependencies in the client job image.

Configure both sites for the `gen3-demo` study using the [multi-study guide](../../../docs/user_guide/admin_guide/multi_study_guide.rst). Each site supplies its own credentials; the server does not need them. The principal's permissions are configured in Gen3, independently of FLARE job submission.

## Run

Export the Gen3 job:

```bash
python job.py --manifest /tmp/gen3-demo-data/manifest.json \
  --job-dir /tmp/gen3-demo-jobs
```

Submit the exported job to the `gen3-demo` study. The clients download their assigned files with the Gen3 SDK, load NumPy arrays, and perform three federated rounds of linear regression.

For a local simulator control using the same files:

```bash
GEN3_LOCAL_DATA_DIR=/tmp/gen3-demo-data python job.py \
  --manifest /tmp/gen3-demo-data/manifest.json \
  --source local --workspace /tmp/gen3-demo-control
```

Check the saved server model against the independent calculation:

```bash
python verify_result.py --data /tmp/gen3-demo-data \
  --model /path/to/server/model.npy
```

Use `--control /path/to/control/model.npy` to compare with a pre-staged run. Use matching `--rounds` values when changing the default round count.

Downloads use the SDK's standard behavior. Data stays in the job's temporary directory and in memory during training; subsequent rounds reuse it.

## Tests

From the repository root:

```bash
python -m pytest -q examples/advanced/gen3/tests
```

See the [Gen3 file SDK documentation](https://uc-cdis.github.io/gen3sdk-python/_build/html/file.html) for the download API.
