# Example code

The runnable Python files are in the chapter's shared [code directory](../../code):

- [job.py](../../code/job.py): build and run the federated learning recipe.
- [client.py](../../code/client.py): train and evaluate at each client.
- [model.py](../../code/model.py): define the PyTorch model.

After completing [setup](../setup.ipynb), run these commands from this directory:

```bash
cd ../../code
python job.py
```

All lessons use that shared project. The former per-lesson `src` files have moved there; local Jupyter checkpoints can leave the old directories visible after switching branches.
