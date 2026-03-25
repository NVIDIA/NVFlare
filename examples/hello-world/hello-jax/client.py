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

"""
Client-side JAX/Flax training script for the hello-jax example.
"""

import argparse
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
from model import MODEL, create_train_state, flatten_params, unflatten_params

import nvflare.client as flare
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.np.constants import NPConstants


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_partitions", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="/tmp/nvflare/data/hello-jax/mnist")
    return parser.parse_args()


def load_mnist(data_dir: str):
    train_images = np.load(f"{data_dir}/train_images.npy").astype(np.float32) / 255.0
    train_labels = np.load(f"{data_dir}/train_labels.npy").astype(np.int32)
    test_images = np.load(f"{data_dir}/test_images.npy").astype(np.float32) / 255.0
    test_labels = np.load(f"{data_dir}/test_labels.npy").astype(np.int32)

    return (train_images, train_labels), (test_images, test_labels)


def split_for_client(images, labels, client_name: str, num_partitions: int):
    match = re.search(r"(\d+)$", client_name)
    if not match:
        return images, labels

    client_idx = max(int(match.group(1)) - 1, 0)
    partitions = max(num_partitions, 1)
    image_splits = np.array_split(images, partitions)
    label_splits = np.array_split(labels, partitions)
    if client_idx >= len(image_splits):
        return images, labels
    return image_splits[client_idx], label_splits[client_idx]


@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        logits = MODEL.apply({"params": params}, images)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, accuracy


@jax.jit
def eval_step(params, images, labels):
    logits = MODEL.apply({"params": params}, images)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, accuracy


def train_epoch(state, images, labels, batch_size: int, rng):
    num_examples = len(images)
    if num_examples == 0:
        raise ValueError("No training data available for this client.")

    permutation = np.asarray(jax.random.permutation(rng, num_examples))
    total_loss = 0.0
    total_accuracy = 0.0
    steps = 0

    for start in range(0, num_examples, batch_size):
        end = start + batch_size
        indices = permutation[start:end]
        batch_images = jnp.asarray(images[indices])
        batch_labels = jnp.asarray(labels[indices])
        state, loss, accuracy = train_step(state, batch_images, batch_labels)
        total_loss += float(loss)
        total_accuracy += float(accuracy)
        steps += 1

    return state, total_loss / steps, total_accuracy / steps, steps


def evaluate(params, images, labels, batch_size: int):
    num_examples = len(images)
    if num_examples == 0:
        raise ValueError("No evaluation data available for this client.")

    total_loss = 0.0
    total_accuracy = 0.0
    steps = 0

    for start in range(0, num_examples, batch_size):
        end = start + batch_size
        batch_images = jnp.asarray(images[start:end])
        batch_labels = jnp.asarray(labels[start:end])
        loss, accuracy = eval_step(params, batch_images, batch_labels)
        total_loss += float(loss)
        total_accuracy += float(accuracy)
        steps += 1

    return total_loss / steps, total_accuracy / steps


def main():
    args = parse_args()
    flare.init()

    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    (train_images, train_labels), (test_images, test_labels) = load_mnist(args.data_dir)
    train_images, train_labels = split_for_client(train_images, train_labels, client_name, args.num_partitions)
    test_images, test_labels = split_for_client(test_images, test_labels, client_name, args.num_partitions)

    print(f"site={client_name}, train_samples={len(train_images)}, test_samples={len(test_images)}")

    rng = jax.random.PRNGKey(0)
    while flare.is_running():
        input_model = flare.receive()
        current_round = input_model.current_round
        flat_params = input_model.params[NPConstants.NUMPY_KEY]
        params = unflatten_params(flat_params)

        eval_loss, accuracy = evaluate(params, test_images, test_labels, args.batch_size)
        print(
            f"site={client_name}, round={current_round}, "
            f"received_model_eval_loss={eval_loss:.4f}, accuracy={accuracy:.4f}"
        )

        if flare.is_evaluate():
            flare.send(flare.FLModel(metrics={"accuracy": accuracy, "eval_loss": eval_loss}))
            continue

        state = create_train_state(params, learning_rate=args.learning_rate, momentum=args.momentum)
        steps_per_epoch = math.ceil(len(train_images) / args.batch_size)

        for epoch in range(args.epochs):
            rng, epoch_rng = jax.random.split(rng)
            state, train_loss, train_accuracy, _ = train_epoch(
                state,
                train_images,
                train_labels,
                args.batch_size,
                epoch_rng,
            )
            print(
                f"site={client_name}, round={current_round}, epoch={epoch + 1}, "
                f"train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f}"
            )

        updated_params = flatten_params(state.params)
        output_model = flare.FLModel(
            params={NPConstants.NUMPY_KEY: updated_params},
            params_type=flare.ParamsType.FULL,
            metrics={"accuracy": accuracy, "eval_loss": eval_loss},
            meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: args.epochs * steps_per_epoch},
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()
