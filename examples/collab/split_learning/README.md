# Split Learning with Collab

This is a computation-equivalent Collab rewrite of the original
[CIFAR-10 SplitNN example](../../advanced/vertical_federated_learning/cifar10-splitnn/README.md).
It retains the original training algorithm and defaults while replacing its
controller, executor, `Shareable`, `DXO`, serialization, and auxiliary message
handlers with direct function calls.

- `site-1` owns the CIFAR-10 images and the convolutional half of ModerateCNN.
- `site-2` owns the labels and the fully connected half.
- Raw images and labels remain at their respective sites.
- The sites exchange FP16 training activations and gradients by default;
  validation activations remain FP32, as in the original.

The rewrite uses the same ModerateCNN layers and initialization, CIFAR-10
transforms, random minibatch sampling, SGD configuration, validation batching,
and delayed image-side gradient update as the original. TensorBoard and
low-level timing instrumentation are intentionally omitted because they do not
affect training results.

## The complete training protocol

The original image-side learner applies the gradient returned by `site-2` at
the start of the next round. The Collab workflow preserves that ordering:

```python
gradient = None
for current_round in range(num_rounds):
    if gradient is not None:
        image_site.backward(gradient)
    sample_ids, activations = image_site.forward()
    gradient, loss, accuracy = label_site.compute_gradient(sample_ids, activations)
```

As in the original, the final returned gradient is not applied because there
is no following round. Validation uses all CIFAR-10 validation samples and
`numpy.array_split` with the configured training batch size.

## Data alignment

Both sites must agree on the meaning and ordering of sample IDs. With no
`--intersection-file`, the complete aligned CIFAR-10 training set is used. To
match an original run that first performs private set intersection, pass its
intersection file:

```bash
--intersection-file /path/to/intersection.txt
```

In a real application, align private sample IDs first (for example, with PSI)
and consider the privacy implications of shared activations and gradients.

## Run it

Install the dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

Install NVFlare from this repository, then run from the `examples` directory:

```bash
cd examples
python -m collab.split_learning.split_learning
```

The defaults now match the original job: 15,625 minibatches, batch size 64,
learning rate 0.01, seed 42, FP16 transfer, and full validation every 1,000
rounds. Like the original, this is a substantial training run. For a quick
smoke test without validation:

```bash
python -m collab.split_learning.split_learning \
    --num-rounds 1 --validation-frequency 0
```

The same recipe can use real local FLARE processes or be exported as a job:

```bash
python -m collab.split_learning.split_learning --runtime multi_process
python -m collab.split_learning.split_learning \
    --runtime export --job-root /tmp/jobs
```

Run with `--help` for dataset, intersection, FP16, training, logging, runtime,
and deployment options. This example always requires exactly two clients named
`site-1` and `site-2`.
