# Framework Routing

Read the active training entry point without executing it. Ask: who owns
training?

- Hugging Face: a `transformers.Trainer`, `Seq2SeqTrainer`, or supported TRL
  trainer owns `train()` / `evaluate()`.
- Lightning: a Lightning `Trainer` owns `fit()` / `validate()` / `test()`.
- Plain PyTorch: user code owns a manual batch loop with backward and optimizer
  steps.

Imports do not establish ownership. Cite the file and line, then route to
`nvflare-convert-huggingface`, `nvflare-convert-lightning`, or
`nvflare-convert-pytorch`. If the owner is hidden, conflicting, or differs
between entry points, use `nvflare-orient` to ask which entry point to federate.
