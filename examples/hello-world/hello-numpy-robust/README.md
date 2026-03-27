# Hello NumPy Robust Aggregation

This example extends the NumPy hello-world recipe with a robust aggregation option.

## What it demonstrates

- Standard FedAvg aggregation (`--aggregator default`)
- Byzantine-resilient element-wise median aggregation (`--aggregator median`)
- A simple poisoned-client simulation to compare behavior under outliers

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional: enable TensorBoard tracking (requires PyTorch dependency used by NVFlare tracking receiver):

```bash
python -m pip install torch
```

Run baseline FedAvg with one poisoned client:

```bash
python job.py --aggregator default --n_clients 5 --num_rounds 3 --poison_client_name site-1 --poison_scale 1000
```

Run robust median aggregation with the same setup:

```bash
python job.py --aggregator median --n_clients 5 --num_rounds 3 --poison_client_name site-1 --poison_scale 1000
```

Run with TensorBoard tracking enabled:

```bash
python job.py --aggregator median --tracking tensorboard
```

Disable poisoning:

```bash
python job.py --aggregator median --poison_client_name ""
```

## Expected outcome

With poisoning enabled, the default aggregator can be strongly shifted by the poisoned update,
while the median aggregator remains close to benign client updates.

### Example output snapshot (1 round, 5 clients, `site-1` poisoned with scale 1000)

| Mode | Persisted server model (`server.npy`) |
| --- | --- |
| `--aggregator default` | `[[401.6, 602.4, 803.2], [1004.0, 1204.8, 1405.6], [1606.4, 1807.2, 2008.0]]` |
| `--aggregator median` | `[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]` |

These values are from local simulator runs and show the expected robustness pattern: the median aggregator rejects the extreme outlier influence.

## Tests

```bash
python -m pytest tests/test_custom_aggregators.py -v
```
