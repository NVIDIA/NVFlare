# FedMoSA: Federated Medical Image Segmentation under Modality Heterogeneity via Specialized Adapters

This directory has an NVFlare implementation of FedMoSA (MICCAI 2026). FedMoSA adapts a frozen Segment Anything (SAM ViT-B) model across clients that hold different imaging modalities (for example CT and MR). Every image-encoder block has modality-specialized adapters and shared adapters. The server aggregates each modality expert only over the clients that hold that modality, and aggregates the shared components with modality-balanced weights. Clients align their shared-adapter prototypes to the aggregated global prototypes.

## Links

- Paper: [MICCAI 2026, LNCS, pp. 195–205](https://link.springer.com/chapter/10.1007/978-3-032-38072-2_20)
- Original code: https://github.com/obedjamir/FedMoSA

## Method

For client $k$ with dataset $D_k$, modalities $M_k$ and $n_{k,m}$ samples of modality $m$:

| Component | Aggregation weight |
| --- | --- |
| Expert of modality $m$ (`experts.<idx(m)>`) | $n_{k,m} / \sum_j n_{j,m}$, only over the clients holding $m$ |
| Shared adapters, mask decoder, no-mask embedding | $\propto \|D_k\| \cdot \|M_k\|$ |
| Shared-adapter prototypes | the same balanced weights |

Local loss: $\lambda_{bce}\,\mathrm{BCE} + \lambda_{dice}\,\mathrm{Dice} + \lambda_{proto}\,\frac{1}{2L}\sum_{l}\sum_{b\in\{attn,mlp\}} \lVert p^{local}_{l,b} - p^{global}_{l,b} \rVert_2$.

As in the original implementation, the adapted blocks apply SAM's attention to the full token grid (no window partitioning) and do not back-propagate through SAM's frozen attention / MLP (the adapters receive gradients through the residual stream). Each batch has one modality and is routed through that modality's experts. The SAM image encoder stays frozen; the mask decoder is trained from a random initialization (`--decoder-init pretrained` starts it from the SAM checkpoint instead). No prompts are used, so of the prompt encoder only its no-mask embedding is trained. The adapters, the mask decoder and the no-mask embedding are exchanged, together with the shared prototypes.

## Code layout

| File | Contents |
| --- | --- |
| `job.py` | Builds the FedJob (custom controller + Client API script) and runs the simulator |
| `src/fedmosa_controller.py` | `FedMoSAController` (a `ModelController`): modality-restricted and modality-balanced aggregation, prototype aggregation |
| `src/client.py` | Client API training script |
| `src/model.py` | MoSA: SAM ViT-B with modality-specialized and shared adapters |
| `src/data.py` | Loading the prepared slices, the original transforms and modality-alternating batches, synthetic data |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Data

The NVFlare example uses exactly the same slices as the original code. Build them once with the original
pipeline of the [original repository](https://github.com/obedjamir/FedMoSA) (from its `system/`, after
`dataset/generate_liver_seg.py`):

```bash
git clone https://github.com/obedjamir/FedMoSA && cd FedMoSA/system
python prepare_data.py -data liver_seg -nc 3
```

This writes `../dataset/liver_seg/slices_<k>_*.pkl` for every client k: all slices of its training volumes and its
val / test slices. The k-th site in `--sites` reads client k−1's files. Every local epoch trains on a fresh random 25% of each
training volume's slices, drawn with the same seed (dataset, run, client, round, epoch) as the original code, so both
implementations see the same subsets. Batching, rotation and transforms follow the original client: one modality
per batch, modalities alternated, each modality shuffled and trimmed to full batches, random 90/180/270° rotation,
1024×1024 inputs and 256×256 masks.

For a quick test with random ellipses:

```bash
python -c "from src.data import make_synthetic; make_synthetic('/tmp/fedmosa_syn', [['CT'], ['CT', 'MR'], ['MR']])"
```

## Run

```bash
python job.py --data-dir /path/to/FedMoSA/dataset/liver_seg --sites site-1,site-2,site-3 \
    --sam-ckpt sam_vit_b_01ec64.pth --num-rounds 50 --gpu 0,1,2
```

From round 1 on, every client logs the val and test IoU / Dice of the global model from the previous round; after
the last round the final global model is evaluated the same way. `--export-only` writes the job folder to
`<workspace>/jobs` for `nvflare job submit` (POC or production).

## Validation

A short functional run of this example on the paper's liver setting with the most heterogeneous split
(`--dataset-id 4`): three MR-only clients and two CT-only clients.

| Item | Value |
| --- | --- |
| Tested version | [`2e45d5e`](https://github.com/obedjamir/FedMoSA/tree/2e45d5e/nvflare) |
| Environment | NVIDIA H200 (140 GB), Python 3.10, NVFlare 2.9.0, simulator with 5 clients |
| Data | 5 clients: MR-only (sites 1–3: 828 / 821 / 777 training slices per epoch) and CT-only (sites 4–5: 4058 / 3927); val / test 354–2253 slices per client |
| Command | `python job.py --data-dir /path/to/FedMoSA/dataset/liver_seg --dataset-id 4 --sites site-1,site-2,site-3,site-4,site-5 --sam-ckpt $(pwd)/sam_vit_b_01ec64.pth --num-rounds 5 --gpu <gpus>` (defaults: 2 local epochs, batch size 8, lr 1e-4) |
| Run time | about 3 h 30 min for 5 rounds plus the final evaluation |
| Observed result | Every round aggregated all 5 clients; the test Dice of the global model rose on every client (table below) |
| Limitations | Functional check with 5 rounds, not a reproduction of the paper's training length; one run. Each client trains a full SAM ViT-B at 1024×1024 (about 30–40 GB of GPU memory at batch size 8). The k-th name in `--sites` reads client k−1's files. |

Test IoU / Dice of the global model (after round 1 = after the first aggregation; final = after round 5):

| Site | Modality | After round 1 | After round 3 | Final |
| --- | --- | --- | --- | --- |
| site-1 | MR | 0.459 / 0.621 | 0.624 / 0.761 | 0.670 / 0.798 |
| site-2 | MR | 0.568 / 0.722 | 0.729 / 0.840 | 0.776 / 0.872 |
| site-3 | MR | 0.556 / 0.712 | 0.696 / 0.816 | 0.737 / 0.845 |
| site-4 | CT | 0.640 / 0.777 | 0.747 / 0.852 | 0.799 / 0.887 |
| site-5 | CT | 0.680 / 0.805 | 0.767 / 0.865 | 0.808 / 0.891 |

## Results

The paper's results (MICCAI 2026) were obtained with the original implementation in the original repository,
trained for the full number of rounds (https://github.com/obedjamir/FedMoSA); see the paper for the comparison with other methods. The run above validates
this NVFlare integration and is not a replacement for those results.

## License

Apache 2.0. SAM weights are under the Segment Anything license.

## Citation

```bibtex
@inproceedings{jamir2027fedmosa,
  title     = {Federated Medical Image Segmentation under Modality Heterogeneity via Specialized Adapters},
  author    = {Jamir, Obed and Paul, Angshuman},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  publisher = {Springer},
  address   = {Cham, Switzerland},
  pages     = {195--205},
  year      = {2027},
  doi       = {10.1007/978-3-032-38072-2_20}
}
```
