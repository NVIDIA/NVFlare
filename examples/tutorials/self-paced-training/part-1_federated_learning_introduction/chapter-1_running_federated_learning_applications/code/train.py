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

"""Run ordinary local training: python train.py."""

from model import create_model
from training import evaluate, make_loaders, train, training_parser


def main():
    args = training_parser().parse_args()
    model = create_model()
    train_loader, eval_loader = make_loaders("site-1", args.batch_size)
    before = evaluate(model, eval_loader)
    steps, loss = train(model, train_loader, args.local_epochs, args.learning_rate)
    after = evaluate(model, eval_loader)
    print(f"Local training: {steps} optimizer steps, loss={loss:.4f}, accuracy={before:.1f}% -> {after:.1f}%")


if __name__ == "__main__":
    main()
