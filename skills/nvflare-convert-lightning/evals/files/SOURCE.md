# Fixture Source Notes

The `hello-lightning` fixtures are minimized, unconverted PyTorch Lightning
training code modeled on the NVFLARE repository example:

- Source example: `examples/hello-world/hello-lightning`

The fixture intentionally omits real datasets, data download, FLARE integration,
and full job execution details so trigger and behavior evals stay deterministic.
`train.py` and `model.py` represent plain Lightning code before any FLARE
conversion; the agent under evaluation is expected to add the
`flare.patch(trainer)` Client API integration and a `job.py`.

The `vocab-lightning` fixture adds a `LitTextCNN` model whose `__init__` has a
required, data-derived argument (`vocab_size`, no default). The conversion must
pin one shared vocabulary size for the server recipe model config and every
client model construction path. Passing a live `LightningModule` instance with
required args can serialize without those args and fail server-side
reconstruction in the model persistor.

The `imbalanced-lightning` fixture is a single-node binary classifier that
derives per-example loss scaling from the labels in the training data. The
conversion should keep that statistic local to each site's training partition
and should not convert it into a globally pinned model or recipe constructor
value.
