# FedMoSA: Federated Modality-Specialized Adapters for Multi-Modal Medical Image Segmentation

This directory has an NVFlare implementation of FedMoSA (MICCAI 2026). FedMoSA adapts a frozen Segment Anything (SAM ViT-B) model across clients that hold different imaging modalities (for example CT and MR). Every image-encoder block has modality-specialized adapters and shared adapters. The server aggregates each modality expert only over the clients that hold that modality, and aggregates the shared components with modality-balanced weights. Clients align their shared-adapter prototypes to the aggregated global prototypes.

## Links

- Paper: MICCAI 2026 (LNCS, pp. 195–205)
- Original code: https://github.com/obedjamir/FedMoSA

## Method

For client $k$ with dataset $D_k$, modalities $M_k$ and $n_{k,m}$ samples of modality $m$:

| Component | Aggregation weight |
| --- | --- |
| Expert of modality $m$ (`experts.<idx(m)>`) | $n_{k,m} / \sum_j n_{j,m}$, only over the clients holding $m$ |
| Shared adapters, prompt encoder, mask decoder | $\propto \|D_k\| \cdot \|M_k\|$ |
| Shared-adapter prototypes | the same balanced weights |

Local loss: $\lambda_{bce}\,\mathrm{BCE} + \lambda_{dice}\,\mathrm{Dice} + \lambda_{proto}\,\frac{1}{2L}\sum_{l}\sum_{b\in\{attn,mlp\}} \lVert p^{local}_{l,b} - p^{global}_{l,b} \rVert_2$.

Each batch has one modality and is routed through that modality's experts. The SAM backbone stays frozen. Only the adapters, the prompt encoder and the mask decoder are exchanged.

## Code layout

| File | Contents |
| --- | --- |
| `job.py` | Builds the FedJob (custom controller + Client API script) and runs the simulator |
| `src/fedmosa_controller.py` | `FedMoSAController` (a `ModelController`): modality-restricted and modality-balanced aggregation, prototype aggregation |
| `src/client.py` | Client API training script |
| `src/model.py` | MoSA: SAM ViT-B with modality-specialized and shared adapters |
| `src/data.py` | Slice dataset, modality-grouped batches, synthetic data for smoke tests |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Data

Each site reads `<data-root>/<site>/{train,test}.csv` with the columns `image,mask,modality`. Each row points to 2D `.npy` slices: the image is normalized to [0, 1] and the mask is binary. The modality must be one of `--modalities`. The paper's liver setting (CHAOS CT/MR, LiTS, …) can be prepared with the preprocessing scripts of the original repository and exported to this format.

For a quick test with random ellipses:

```bash
python -c "from src.data import make_synthetic; make_synthetic('/tmp/fedmosa_syn', {'site-1': ['CT'], 'site-2': ['CT', 'MR'], 'site-3': ['MR']})"
```

## Run

```bash
python job.py --data-root /tmp/fedmosa_syn --sites site-1,site-2,site-3 \
    --sam-ckpt sam_vit_b_01ec64.pth --num-rounds 50 --batch-size 4 --gpu 0,1,2 --threads 3
```

`--export-only` writes the job folder to `<workspace>/jobs` for `nvflare job submit` (POC or production). Each client logs the test Dice of the global model and of the locally trained model every round.

## Validation

| Item | Value |
| --- | --- |
| Tested commit | TODO |
| Environment | TODO (GPU, CUDA, Python) |
| NVFlare version | 2.9.0 |
| Command | TODO |
| Observed result | TODO |
| Limitations | One SAM ViT-B per client at 1024×1024 needs about 4–5 GB in memory, so use a GPU per client. |

## Results

TODO: the test Dice per client from the validation run, compared with the paper.

## License

Apache 2.0. SAM weights are under the Segment Anything license.

## Citation

```bibtex
@inproceedings{jamir2027fedmosa,
  title     = {FedMoSA: Federated Modality-Specialized Adapters for Multi-Modal Medical Image Segmentation},
  author    = {Jamir, Obed and Paul, Angshuman},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  pages     = {195--205},
  year      = {2027},
  publisher = {Springer}
}
```
