.. _memorization_reduction:

Memorization Reduction (Experimental)
======================================

The PyTorch Target Reference Advantage Penalty (TRAP) changes the local training
objective while retaining ordinary FedAvg or FedOpt aggregation. It penalizes
observed-token log probabilities that exceed a frozen reference by a configured
threshold. It is an experimental mitigation, with no differential privacy guarantee.

The method is introduced by Muhammed Ustaomeroglu et al. (2026) in
`TRAP: Understanding and Mitigating Privacy Memorization in Language Models
<https://arxiv.org/abs/2609.32293>`_, arXiv:2609.32293. The components below
integrate the preprint's reference-based penalty into NVFlare.

Loss and diagnostics
--------------------

``nvflare.app_opt.pt.trap_loss.PTTRAPLoss`` returns an additive scalar penalty:

.. math::

   L_{\mathrm{TRAP}} = \frac{\lambda}{|M|}\sum_{t\in M}
       \max\left(0, \log p_\theta(x_t\mid x_{<t}) - \log q(x_t\mid x_{<t}) - \epsilon\right).

Use the same valid tokens and normalization for ordinary cross-entropy and TRAP.
``causal_token_log_probs`` shifts causal logits and labels once; mask prompt and
padding positions in the unshifted labels with ``-100``.

.. code-block:: python

    from nvflare.app_opt.pt.trap_loss import PTTRAPLoss, causal_token_log_probs

    trap = PTTRAPLoss(weight=0.5, threshold=0.0)
    log_probs, mask = causal_token_log_probs(target_logits, labels)
    count = mask.sum().clamp_min(1)
    ordinary_ce = -log_probs.masked_select(mask).sum() / count
    loss = ordinary_ce + trap(log_probs, reference_log_probs, mask)
    loss.backward()
    diagnostics = trap.diagnostics(log_probs, reference_log_probs, mask)

The reference is detached from autograd. Empty masks produce a differentiable zero
penalty. With ``weight=0``, the penalty has zero target gradient; callers can also
skip reference scoring entirely. Weights above one can reverse the active
token-level cross-entropy gradient and should be evaluated as experimental settings.
Diagnostics include valid-token count, mean reference advantage, active-token
fraction, and penalty magnitude.

For gradient accumulation, pass the total valid-token count of the entire
accumulation window as ``normalizer`` to each microbatch penalty, normalize its
ordinary cross-entropy sum by that same count, and sum the contributions. Averaging
microbatch means with different valid-token counts changes the objective.

Frozen reference scoring
------------------------

``nvflare.app_opt.pt.reference_scorer.FrozenReferenceScorer`` accepts a separate
causal language model. It freezes parameters, enforces evaluation mode, and computes
observed-token log probabilities without gradients. Hugging Face is optional for
the SDK: the model may return an object with ``logits`` or a logits tensor.

.. code-block:: python

    from nvflare.app_opt.pt.reference_scorer import FrozenReferenceScorer

    reference = FrozenReferenceScorer(
        reference_model,
        tokenizer_fingerprint="sha256-of-tokenizer-assets",
        scoring_identity="model-config-and-context-recipe-v1;fp32",
        cache_dir="local_reference_cache",  # omit for online scoring
    )
    reference_log_probs, reference_mask = reference.score(
        input_ids=input_ids, labels=labels, attention_mask=attention_mask,
    )

The scorer is a separate object, not a submodule of the target model. Keep reference
parameters out of the target optimizer and transmitted state. A same-task reference
that has not trained on the protected records is the intended starting point.
Provenance cannot rule out unknown exposure during pretraining.

Cache identity includes checkpoint tensor contents, tokenizer/scoring identities,
and all supplied tokenized inputs, labels, and tensor model inputs. Callers must
include model configuration, context construction, and scoring settings in the
scoring identity. Cached values are local tensors loaded with ``weights_only=True``.
Treat them as potentially sensitive artifacts.

Keep the reference immutable. Ordinary in-place parameter/buffer changes are detected;
construct a new scorer after a change. Mutating ``.data`` bypasses PyTorch version
tracking and is unsupported. Arbitrary changes to forward behavior also require a
new scoring identity. A fixed-order precomputation, indexed by shuffled training
batches, avoids cache misses caused solely by different batch groupings.

Auditing and integration
------------------------

Use :doc:`filters` for the ``ModelFilter`` interface and the report-only
``MemorizationAuditFilter``. The ``PTMemorizationAuditor`` adapter handles a separate
evaluation module, full-weight/difference reconstruction, membership AUROC/nAUC,
before/after scores, and RNG isolation. Task-specific data and extraction probes
are supplied by the application. ``MemorizationAuditReporter`` persists local
aggregate reports and optionally emits explicitly selected scalar metrics.

Client-model audits measure the transmitted update at the configured filter stage.
They do not replace evaluation of the selected global checkpoint or of exchanged
reference models. Incomplete audits must not be interpreted as low leakage.
External Client API filters require materialized parameters, increasing runtime
memory and transfer cost relative to forwarding lazy references.

The `TRAP research entry point
<https://github.com/NVIDIA/NVFlare/tree/main/research/memorization-reduction>`_
uses a tiny CPU model, local synthetic records, and two-client FedAvg. It exercises
loss integration and periodic audits without downloading a model or dataset.
It is a functional example and supplies no evidence of privacy or utility benefit.

Full paired benchmarking results will be shared separately. The
:doc:`design document <../design/federated_memorization_reduction>`
describes the API boundaries, reference assumptions, resource costs and review gates.

.. toctree::
   :hidden:

   ../design/federated_memorization_reduction
