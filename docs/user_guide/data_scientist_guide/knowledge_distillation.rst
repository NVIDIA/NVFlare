.. _knowledge_distillation:

Standalone PyTorch Knowledge Distillation
=========================================

The utilities in ``nvflare.app_opt.pt.distillation`` distill a student from one
or more teachers over caller-supplied inputs. They can run in a local script,
a client trainer or a server application without starting a FLARE workflow.
PyTorch is required; Collab, datasets and synthesis libraries are not used.

Use the Components
------------------

The loss, history buffer and trainer can be imported individually:

.. code-block:: python

   from functools import partial

   import torch
   from torch import nn

   from nvflare.app_opt.pt.distillation.losses import kl_divergence
   from nvflare.app_opt.pt.distillation.teachers import TeacherBuffer
   from nvflare.app_opt.pt.distillation.trainer import DistillationTrainer

   # supplied_teacher and supplied_batches belong to the application.
   history = TeacherBuffer(max_size=2)
   history.add(supplied_teacher, metadata={"version": 7})

   # Reconstruct teachers with the correct architecture for each snapshot.
   teachers = []
   for snapshot in history:
       model = teacher_factory()
       model.load_state_dict(snapshot.model_state)
       teachers.append(model)

   trainer = DistillationTrainer(
       student=lambda: nn.Linear(input_features, num_classes),
       teachers=teachers,
       optimizer_factory=partial(torch.optim.Adam, lr=1e-3),
       device="cpu",
       batch_adapter=lambda batch: batch[0],  # e.g. (inputs, unused labels)
       loss_fn=partial(kl_divergence, temperature=2.0),
   )
   metrics = trainer.fit(supplied_batches, max_steps=100)
   distilled_student = trainer.student

Different teacher/student architectures are supported when their output classes
align. Parameter-space operations, such as blending a distilled model delta with
a client delta, require compatible parameters and belong to the consuming
algorithm. See the
:github_nvflare_link:`standalone example <examples/advanced/pt_distillation>`
for an executable MLP-to-linear demonstration over application-generated inputs.

Loss and Output Contract
------------------------

``kl_divergence(student_logits, teacher_output, ...)`` computes
``T**2 * KL(teacher || student)``. Both inputs must be floating tensors on the same
device with identical nonempty ``[batch, classes]`` shapes and the same class
ordering. Equal shapes do not verify class semantics; callers must align them.

The student always supplies raw logits. The teacher supplies raw logits by
default. With ``teacher_is_logits=False``, supply normalized probabilities already
softened at the same temperature; the function does not apply temperature a
second time. Teacher values are detached, so gradients reach only the student.

``batchmean`` sums class contributions and divides by batch size. ``sum`` sums
all entries. ``none`` returns individual ``[batch, classes]`` terms, which can
be negative even though their complete KL sum is nonnegative. The trainer
requires a scalar batch-mean loss. Arithmetic follows native PyTorch dtypes;
there is no implicit precision conversion or overflow recovery.

The default batch is an input tensor. ``batch_adapter`` may extract it from
another structure; it is detached and moved to the trainer's device. Wrap models
to handle multiple inputs. ``student_output_adapter`` and
``teacher_output_adapter`` extract/reorder logits from dictionaries, tuples or
other model outputs. The teacher adapter applies to every teacher; wrap teachers
individually when their output conventions differ.

Training and Ownership
----------------------

``fit`` consumes one pass over the supplied iterable, optionally capped by
``max_steps``. It never cycles exhausted data or consumes an extra batch beyond
the cap. Use another ``fit`` call for another epoch. Returned metrics contain
completed steps, samples and a sample-weighted mean loss. An empty input or a
non-finite/non-scalar loss raises; completed steps are not rolled back.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Ownership and behavior
   * - Student
     - A supplied module is moved and updated in place. A zero-argument factory
       must return a fresh module. Supplied weights are retained, without a
       hidden reset. Training puts the student in train mode and leaves it there.
   * - Teachers
     - Modules/factory results are copied once into trainer-owned models, frozen
       and put in eval mode. Original parameters, gradients, buffers, modes and
       devices remain unchanged. Each teacher contributes equally through its
       own loss; teacher logits are not averaged.
   * - TeacherBuffer
     - Stores at most ``max_size`` snapshots, newest first. Adding copies
       parameters/buffers to CPU and independently copies optional metadata.
       Iterated snapshots are borrowed read-only; their tensor/dict contents are
       not immutable. ``state_dict()`` returns independent export copies.
   * - Optimizer
     - Created after device placement by ``optimizer_factory`` and retained
       across ``fit`` calls. No learning-rate schedule or optimizer reset is
       implicit.
   * - Randomness and data
     - The caller owns initialization, dropout, augmentation, shuffling and
       generators. Components do not set global seeds. Factories and training
       can consume RNG; the snapshot buffer itself performs no random sampling.

Component State
---------------

``TeacherBuffer.state_dict()`` exports capacity, model states and metadata.
Restore into a buffer with matching capacity; entries are never silently dropped.
The buffer stores no model factories or architecture objects. Metadata can carry
an application architecture identifier when snapshots come from different models.

``DistillationTrainer.state_dict()`` copies student/teacher weights, optimizer
state and cumulative step count. Reconstruct matching architectures, teacher
order, optimizer, adapters and loss configuration before loading. Model states
are CPU copies; optimizer state follows PyTorch's native format/devices. For
example, the application can save CPU RNG alongside the component state:

.. code-block:: python

   torch.save(
       {"trainer": trainer.state_dict(), "rng": torch.get_rng_state()},
       checkpoint_path,
   )

   saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
   restored = make_trainer()  # same configuration; constructors can consume RNG
   restored.load_state_dict(saved["trainer"])
   torch.set_rng_state(saved["rng"])  # restore after construction
   restored.fit(next_batches)

Restore any CUDA RNG, data-loader generator/sampler and augmentation state
separately when used. Factories, adapters and loss callables are not serialized.
Metadata must be compatible with the application's chosen serialization format.
Component restoration does not restore distributed assignments or workflow state.

Synthesis providers, synthetic pools, class-aware teacher weighting and
FedRevive composition are separate follow-up components. None are imported by
these basic KD utilities.
