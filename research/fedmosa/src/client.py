# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FedMoSA client (NVFlare Client API): local training with BCE + Dice + shared-prototype alignment."""

import argparse
import os
import sys

import torch

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import batches, epoch_subset, load_split, set_seed, subset_counts, train_seed  # noqa: E402
from src.model import MoSA  # noqa: E402


def dice_loss(logits, target, smooth=1.0):
    """Dice over the whole batch, as in the original client."""
    p = torch.sigmoid(logits).view(-1)
    t = target.view(-1)
    inter = (p * t).sum()
    return 1 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)


def proto_loss(model, global_protos):
    terms = []
    for local, glob in zip(model.current_prototypes(), global_protos):
        for k, g in glob.items():
            terms.append(torch.norm(local[k] - torch.as_tensor(g, device=local[k].device), p=2))
    return torch.stack(terms).mean()


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    """Mean over batches of the batch IoU and Dice, as in the original client."""
    if not data:
        return None
    model.eval()
    ious, dices = [], []
    for x, y, m in batches(data, batch_size, threshold="gt"):
        model.set_modality(m)
        y = y.to(device)
        p = (torch.sigmoid(model(x.to(device))) > 0.5).float()
        inter = (p * y).sum()
        ious.append((inter / torch.clamp(p.sum() + y.sum() - inter, min=1.0)).item())
        dices.append((2.0 * inter / torch.clamp(p.sum() + y.sum(), min=1.0)).item())
    return {"iou": sum(ious) / len(ious), "dice": sum(dices) / len(dices)}


def fmt(r):
    return "n/a" if r is None else f"IoU {r['iou']:.4f} Dice {r['dice']:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dataset folder with the slices_*.pkl files")
    ap.add_argument("--dataset-id", default="0")
    ap.add_argument("--sam-ckpt", default=None)
    ap.add_argument("--modalities", default="MR,CT")
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-bce", type=float, default=0.5)
    ap.add_argument("--lam-dice", type=float, default=0.5)
    ap.add_argument("--lam-proto", type=float, default=0.01)
    args = ap.parse_args()

    flare.init()
    site = flare.get_site_name()
    client_idx = int(site.split("-")[-1]) - 1  # site-1 -> client 0
    set_seed(32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoSA(args.sam_ckpt or None, args.modalities.split(",")).to(device)
    dataset = os.path.basename(os.path.normpath(args.data_dir))  # e.g. liver_seg (part of the subset seed)
    volumes = load_split(args.data_dir, client_idx, "train", args.dataset_id)
    val = load_split(args.data_dir, client_idx, "val", args.dataset_id)
    test = load_split(args.data_dir, client_idx, "test", args.dataset_id)
    counts = {m: 0 for m in model.modalities}
    counts.update(subset_counts(volumes))  # slices per epoch (a fresh 25% of every training volume)
    print(f"[{site}] client {client_idx}: train {counts}, val {len(val or [])}, test {len(test or [])}", flush=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()

    while flare.is_running():
        in_model = flare.receive()
        model.load_state_dict(in_model.params, strict=False)
        rnd = in_model.current_round

        if flare.is_evaluate() or rnd > 0:  # the global model after the previous round's aggregation
            v, t = evaluate(model, val, args.batch_size, device), evaluate(model, test, args.batch_size, device)
            which = "final global model" if flare.is_evaluate() else f"global model after round {rnd - 1}"
            print(f"[{site}] {which}: val {fmt(v)} | test {fmt(t)}", flush=True)
            metrics = {f"{s}_{k}": val_ for s, r in (("val", v), ("test", t)) if r for k, val_ in r.items()}
            if flare.is_evaluate():
                flare.send(FLModel(metrics=metrics))
                continue

        global_protos = (in_model.meta or {}).get("prototypes")
        model.train()
        model.reset_prototypes()
        steps = 0
        for epoch in range(args.local_epochs):
            train = epoch_subset(volumes, train_seed(dataset, args.dataset_id, client_idx, rnd, epoch))
            for x, y, m in batches(train, args.batch_size, threshold="ge"):
                model.set_modality(m)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = args.lam_bce * bce(logits, y) + args.lam_dice * dice_loss(logits, y)
                if global_protos:
                    loss = loss + args.lam_proto * proto_loss(model, global_protos)
                opt.zero_grad()
                loss.backward()
                opt.step()
                steps += 1

        protos = [{k: v.cpu().tolist() for k, v in p.items()} for p in model.mean_prototypes()]  # meta: plain lists
        flare.send(
            FLModel(
                params=model.trainable_state(),
                meta={"mod_counts": counts, "prototypes": protos, "NUM_STEPS_CURRENT_ROUND": steps},
            )
        )


if __name__ == "__main__":
    main()
