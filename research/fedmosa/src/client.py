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
import torch.nn.functional as F

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import SliceDataset, modality_batches, modality_counts, read_rows  # noqa: E402
from src.model import MoSA  # noqa: E402


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


def proto_loss(model, global_protos):
    terms = []
    for local, glob in zip(model.current_prototypes(), global_protos):
        for k, g in glob.items():
            terms.append(torch.norm(local[k] - torch.as_tensor(g, device=local[k].device), p=2))
    return torch.stack(terms).mean()


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    model.eval()
    dices = []
    for x, y, m in modality_batches(dataset, batch_size, shuffle=False):
        model.set_modality(m)
        p = (torch.sigmoid(model(x.to(device))) > 0.5).float()
        y = y.to(device)
        inter = (p * y).sum((1, 2, 3))
        dices += ((2 * inter + 1e-6) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1e-6)).tolist()
    return sum(dices) / max(len(dices), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--sam-ckpt", default=None)
    ap.add_argument("--modalities", default="CT,MR")
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-bce", type=float, default=1.0)
    ap.add_argument("--lam-dice", type=float, default=1.0)
    ap.add_argument("--lam-proto", type=float, default=0.1)
    args = ap.parse_args()

    flare.init()
    site = flare.get_site_name()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoSA(args.sam_ckpt or None, args.modalities.split(",")).to(device)
    train_set = SliceDataset(read_rows(args.data_root, site, "train"), train=True)
    test_set = SliceDataset(read_rows(args.data_root, site, "test"), train=False)
    counts = modality_counts(train_set.rows)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    while flare.is_running():
        in_model = flare.receive()
        model.load_state_dict(in_model.params, strict=False)
        global_protos = (in_model.meta or {}).get("prototypes")
        dice_global = evaluate(model, test_set, args.batch_size, device)

        model.train()
        model.reset_prototypes()
        steps = 0
        for _ in range(args.local_epochs):
            for x, y, m in modality_batches(train_set, args.batch_size):
                model.set_modality(m)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = args.lam_bce * F.binary_cross_entropy_with_logits(logits, y) + args.lam_dice * dice_loss(
                    logits, y
                )
                if global_protos:
                    loss = loss + args.lam_proto * proto_loss(model, global_protos)
                opt.zero_grad()
                loss.backward()
                opt.step()
                steps += 1

        dice_local = evaluate(model, test_set, args.batch_size, device)
        print(
            f"[{site}] round {in_model.current_round}: test Dice global {dice_global:.4f} -> local {dice_local:.4f}",
            flush=True,
        )
        protos = [{k: v.cpu().tolist() for k, v in p.items()} for p in model.mean_prototypes()]  # meta: plain lists
        flare.send(
            FLModel(
                params=model.trainable_state(),
                metrics={"dice": dice_global},
                meta={"mod_counts": counts, "prototypes": protos, "NUM_STEPS_CURRENT_ROUND": steps},
            )
        )


if __name__ == "__main__":
    main()
