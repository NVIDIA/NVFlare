# Standalone PyTorch knowledge distillation

Train a small linear student from two snapshots of an MLP teacher over supplied
proxy inputs. This runs locally with ordinary PyTorch: no FLARE server, client,
Collab session, dataset download or synthesis package is needed.

```bash
pip install -r requirements.txt
python distill.py
```

The script trains the teacher on application-generated classification inputs,
captures two CPU snapshots, and distills a student with a different architecture.
It prints the initial/final mean teacher KL and checks that it decreases.
The labels train the teacher; the distillation loop uses only inputs.
This is a functional example, not a dataset accuracy benchmark.

Import the pieces independently:

```python
from nvflare.app_opt.pt.distillation.losses import kl_divergence
from nvflare.app_opt.pt.distillation.teachers import TeacherBuffer
from nvflare.app_opt.pt.distillation.trainer import DistillationTrainer
```

Use your own batches, model factories, optimizer factory and output adapters.
Changing the teacher list enables single/multiple teacher ablations. Omitting
the trainer leaves the loss and snapshot buffer usable on their own. Models must
agree on output classes, but their parameter shapes may differ.

See the [KD usage guide](../../../docs/user_guide/data_scientist_guide/knowledge_distillation.rst)
for ownership, output/probability conventions and component checkpointing.
