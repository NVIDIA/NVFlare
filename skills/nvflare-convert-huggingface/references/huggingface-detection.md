# Hugging Face Trainer Detection

Use static evidence to decide whether this converter owns the request.

## Positive Evidence

Strong evidence includes:

- construction or subclassing of `transformers.Trainer`,
  `transformers.Seq2SeqTrainer`, TRL `SFTTrainer`, or another documented
  `Trainer` subclass;
- `TrainingArguments`, `Seq2SeqTrainingArguments`, or `SFTConfig` passed to that
  Trainer;
- Trainer-owned `train()` and `evaluate()` calls;
- Trainer callbacks, `compute_metrics`, datasets, processors/tokenizers, data
  collators, or a PEFT config connected to that Trainer;
- an existing `nvflare.client.hf.patch(trainer)` call.

Imports alone are supporting evidence. Do not classify an inference pipeline,
tokenizer utility, dataset preparation script, or isolated `AutoModel` import
as Trainer-style training without examining its entrypoint.

## Routing Boundaries

- Route `AutoModel` plus a manual optimizer/backward loop to
  `nvflare-convert-pytorch`.
- Route `LightningModule` or Lightning `Trainer` ownership to
  `nvflare-convert-lightning`, even when the module contains a Transformers
  model.
- Keep pure preprocessing, model download, inference, evaluation reporting,
  serving, and deployment outside this conversion skill.
- Treat Accelerate-only custom loops as manual PyTorch unless a supported
  Hugging Face `Trainer` still owns training.
- Route a Trainer/configuration factory that static inspection cannot resolve
  to `nvflare-orient` rather than guessing Trainer ownership.

## Facts To Extract

Record exact source locations for:

- Trainer class and construction;
- model class/checkpoint and constructor keyword arguments;
- tokenizer, processor, data collator, formatting function, and dataset fields;
- `TrainingArguments`/`SFTConfig`, especially epochs, steps, output directory,
  saving, evaluation, precision, DeepSpeed, FSDP, and reporting;
- `compute_metrics` output keys and whether higher or lower is better;
- PEFT model/config, adapter type, target modules, rank, alpha, dropout, bias,
  task type, and modules-to-save;
- distributed initialization and launcher command;
- source data paths, site partitioning, and validation split;
- callbacks, trackers, checkpoint resume, and prebuilt optimizer/scheduler.

Never import or execute the training module to discover these facts.
