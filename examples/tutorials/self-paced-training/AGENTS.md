# Self-paced course examples

- Optimize for learners opening and running a section independently. Each hands-on section keeps its own `code/` directory beside its notebook, including its own supporting Python modules.
- Repeating model, data, or training code across sections is intentional. Do not consolidate examples into a parent directory or make one section import another section's implementation.
- Use `job.py`, `client.py`, and `model.py` for the main example files. Keep each section's implementation focused on its own lesson; include additional variants only when that lesson compares them.
- Notebook commands and source displays must use that section's own code. Verify the example runs and exports from its local directory.
- Training and export commands needed by later cells must be executable notebook cells using the selected kernel's Python. Keep a compact exit-status check so a failed command stops execution before later cells can read stale output. Validate notebooks from a fresh workspace without running those commands separately beforehand.
- Keep learner code focused on the APIs taught by each lesson. Put checkpoint assertions and result-integrity checks in validation, and use the printed run summary in introductory lessons. Teach file inspection where it is the lesson's subject.
- Avoid chapter and section numbers in headings and navigation labels so lessons can be reordered.
