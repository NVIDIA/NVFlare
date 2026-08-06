# Split Learning on CIFAR-10 with the Collab API

This example implements the same two-party SplitNN setup as the existing
[CIFAR-10 SplitNN example](../../vertical_federated_learning/cifar10-splitnn/README.md),
using the Collab API:

- `site-1` owns the images and the convolutional bottom model;
- `site-2` owns the labels and the classifier top model;
- raw images and labels remain local;
- site 1 sends aligned batch positions and cut-layer activations, while site 2
  returns cut-layer gradients and training metrics.

The image-side coordinator expresses one training step directly:

```python
gradients = None
for batch_indices in training_batches:
    if gradients is not None:
        self.backward(gradients)
    activations = self.forward(batch_indices)
    gradients, metrics = label_client.compute_loss(batch_indices, activations)
```

Tensors, indices, and metrics remain native Python and PyTorch values in the
application. There are no application-level `Shareable`, `DXO`, data-kind,
header, tensor-decomposer, or auxiliary-message conversions.

## Install

From `examples/advanced/collab/pt_splitnn`, install the example dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the example against the NVFlare installation from this repository.

## Prepare data

Download CIFAR-10 and prepare 10,000 aligned training indices:

```bash
python prepare_data.py \
    --data-root /tmp/nvflare/datasets/cifar10 \
    --overlap 10000
```

Preparation downloads or verifies both CIFAR-10 splits, then writes:

```text
/tmp/nvflare/datasets/cifar10/
├── cifar-10-batches-py/
├── splitnn_intersection.npy
└── splitnn_manifest.json
```

The prepared intersection represents the aligned output that the training
stage consumes. In a deployment where the parties do not already know their
common sample IDs, use a private set intersection workflow before SplitNN
training, as demonstrated by the existing vertical FL example.

Use `--overwrite` when intentionally regenerating existing SplitNN metadata.

## Run

Run the two-site simulation with the settings used by the existing example:

```bash
python job.py \
    --data-root /tmp/nvflare/datasets/cifar10 \
    --num-steps 15625 \
    --batch-size 64 \
    --validation-frequency 1000
```

For a short functional run:

```bash
CUDA_VISIBLE_DEVICES=0 python job.py \
    --data-root /tmp/nvflare/datasets/cifar10 \
    --num-steps 10 \
    --validation-frequency 0 \
    --device cuda
```

Both simulator sites share the single visible GPU in this command. Use
`--device cpu` instead to run without a GPU.

Activations and gradients use float16 for exchange by default, matching the
bandwidth-saving behavior of the existing example. Pass `--fp32` to keep them
in float32.

## Apples-to-apples comparison

Both implementations were run from clean simulator workspaces with the same
10,000-sample intersection, model initialization, random batches, 15,625
steps, batch size 64, SGD parameters, validation cadence, and float16 cut-layer
exchange.

### Benchmark environment

| Component | Configuration |
| --- | --- |
| GPU | One NVIDIA RTX 6000 Ada Generation, 49,140 MiB; both sites shared `cuda:0` |
| NVIDIA driver | 580.173.02 |
| CUDA and cuDNN | PyTorch CUDA 12.6; cuDNN 9.5.1 |
| CPU | Intel Core i7-6800K; 6 cores, 12 threads |
| Host memory | 109 GiB |
| OS | Ubuntu 24.04.4 LTS; Linux kernel 6.8.0-136-generic |
| Python | 3.10.0 |
| PyTorch | 2.6.0+cu126 |
| torchvision | 0.21.0+cu126 |
| NVFlare | Source checkout based on revision `f85d0cb1`, plus this example |

After preparing the data and configuring the intersection as shown in the
[existing SplitNN notebook](../../vertical_federated_learning/cifar10-splitnn/cifar10_split_learning.ipynb),
the existing implementation was launched with:

```bash
cd examples/advanced/vertical_federated_learning/cifar10-splitnn
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from nvflare import SimulatorRunner

simulator = SimulatorRunner(
    job_folder="jobs/cifar10_splitnn",
    workspace="/tmp/nvflare/cifar10_splitnn",
    n_clients=2,
    threads=2,
    log_config="ERROR",
)
simulator.run()
PY
```

The equivalent Collab API run was launched from this example directory with:

```bash
cd examples/advanced/collab/pt_splitnn
CUDA_VISIBLE_DEVICES=0 python job.py \
    --data-root /tmp/nvflare/datasets/cifar10 \
    --num-steps 15625 \
    --batch-size 64 \
    --learning-rate 0.01 \
    --validation-frequency 1000 \
    --device cuda \
    --seed 42
```

| Implementation | Elapsed time |
| --- | ---: |
| Existing SplitNN | 48m 57.7s |
| Collab API SplitNN | 43m 55.9s |

Elapsed time covers each complete command, including simulator startup,
validation, model export, and shutdown. These are single measurements rather
than averages over repeated runs.

The Collab API run was about 10.3% faster in this single-GPU simulator run.
All 15,625 recorded training-loss and training-accuracy values were identical.
The unsmoothed validation curves also align; occasional accuracy differences
were limited to one prediction among the 10,000 validation images.

![Existing and Collab API SplitNN curve alignment](figs/splitnn_curve_alignment.png)

The training panels show a 200-step moving average for readability. The
validation panels show every recorded value without smoothing.

## Outputs

The default simulation workspace is
`/tmp/nvflare/collab/collab_pt_splitnn`. The `site-1` run directory contains:

- `final_model/splitnn_model.pt`, with the trained bottom and top model states;
- `tensorboard/`, with training and validation loss and accuracy.

View the metrics with:

```bash
tensorboard --logdir /tmp/nvflare/collab/collab_pt_splitnn
```

## Why the Collab implementation is simpler

In the component-based implementation, split learning spans several connected
extension points and representations:

1. `SplitNNController` constructs named tasks and their headers.
2. `SplitNNLearnerExecutor` maps those task names into learner methods.
3. Server and client JSON connect the controller, executor, learner,
   persistor, shareable generator, component IDs, and task names.
4. The learner registers label-side handlers under matching auxiliary-message
   topics.
5. Site 1 packages activations and batch indices into `DXO` and `Shareable`
   objects with data kinds, metadata, headers, and cookies, then explicitly
   serializes the tensors with FOBS.
6. Site 2 must validate the same fields, deserialize the activation, compute
   the gradient, and perform the reverse conversion for the reply.

Every producer, consumer, message field, and handler must agree. A mismatch at
any connection can break the algorithm.

With Collab API, the server workflow makes one long-running call to the image
site. The complete SplitNN loop is then implemented in one place in
`trainer.py`: site 1 directly uses the label site's return value as the input
to its next backward pass. The forward/backward dependency, including the
reference implementation's one-step-delayed bottom-model update, is visible as
normal Python control flow.

The remaining distributed concerns are explicit and small: methods crossing a
site boundary are published, the peer site and call timeout are selected, and
tensors are moved to CPU before crossing that boundary. NVFlare handles the
transport representation and job orchestration. This is a one-stop
implementation in the practical sense: the algorithm does not depend on a
separate controller, executor, handler, serializer, and configuration all
agreeing on a parallel protocol. A returned tensor or metric can be reasoned
about at its call site, like any other Python value.

Split learning does not by itself prevent activations, gradients, aligned
sample positions, or other exchanged values from revealing private
information. Apply the privacy protections required by the target deployment.
