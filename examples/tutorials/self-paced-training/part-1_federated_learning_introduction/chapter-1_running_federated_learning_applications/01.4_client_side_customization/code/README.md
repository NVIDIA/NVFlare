# Customize client training

Set different learning rates and batch sizes for site-1 and site-2 in `job.py`.

Complete the course [setup](../../01.1_running_federated_learning_job/setup.ipynb), then run from this `code` directory:

```bash
python job.py
```

This section is self-contained: [job.py](job.py) defines its Recipe, [client.py](client.py) runs at each site, and [model.py](model.py) and [prepare_data.py](prepare_data.py) define its model and synthetic dataset. The training and evaluation functions are directly in the client script. Edit these files to explore this lesson's variation.

The default runtime workspace is `/tmp/nvflare/simulation/chapter1-custom-clients`. Set `NVFLARE_SIMULATOR_WORKSPACE_ROOT` to use a different parent directory, and give the notebook kernel the same value. Exported jobs go under `./exported` when using `--export --export-dir ./exported`.
