# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse

import tensorflow as tf
from data.cifar10_data_utils import load_site_data, preprocess_dataset
from model import ModerateTFNet
from tensorflow.keras import losses

import nvflare.client as flare
from nvflare.app_common.app_constant import AlgorithmConstants
from nvflare.app_opt.tf.scaffold import ScaffoldCallback, TFScaffoldHelper, get_lr_values

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train_idx_root", type=str, default="/tmp/cifar10_splits")
    parser.add_argument("--clip_norm", type=float, default=1.55)
    args = parser.parse_args()

    # Initialize NVFlare client API
    flare.init()
    site_name = flare.get_site_name()

    # Load site-specific data
    train_images, train_labels, test_images, test_labels = load_site_data(site_name, args.train_idx_root)

    # Convert to datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    # Preprocessing
    train_ds = preprocess_dataset(train_ds, is_training=True, batch_size=args.batch_size)
    test_ds = preprocess_dataset(test_ds, is_training=False, batch_size=args.batch_size)

    # Create model
    model = ModerateTFNet()
    model.build(input_shape=(None, 32, 32, 3))

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir="./logs/epochs", write_graph=False)]
    tf_summary_writer = tf.summary.create_file_writer(logdir="./logs/rounds")

    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, clipnorm=args.clip_norm)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.summary()

    # Initialize SCAFFOLD helper
    scaffold_helper = TFScaffoldHelper()
    scaffold_helper.init(model=model)

    while flare.is_running():
        # Receive model from server
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # Load model weights
        for k, v in input_model.params.items():
            model.get_layer(k).set_weights(v)

        # Load SCAFFOLD global controls
        global_ctrl_weights = input_model.meta.get(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL)
        if not global_ctrl_weights:
            raise ValueError("global_ctrl_weights were empty!")

        scaffold_helper.load_global_controls(weights=global_ctrl_weights)
        c_global_para, c_local_para = scaffold_helper.get_params()

        # Clone global model for SCAFFOLD
        model_global = tf.keras.models.clone_model(model)
        model_global.set_weights(model.get_weights())

        # Evaluate global model
        _, test_global_acc = model.evaluate(x=test_ds, verbose=2)

        with tf_summary_writer.as_default():
            tf.summary.scalar("global_model_accuracy", test_global_acc, input_model.current_round)
        print(f"Accuracy of the received model on round {input_model.current_round}: {test_global_acc * 100:.2f}%")

        # Train
        start_epoch = args.epochs * input_model.current_round
        end_epoch = start_epoch + args.epochs

        print(f"Train from epoch {start_epoch} to {end_epoch}")
        model.fit(
            x=train_ds,
            epochs=end_epoch,
            validation_data=test_ds,
            callbacks=callbacks + [ScaffoldCallback(scaffold_helper)],
            initial_epoch=start_epoch,
            validation_freq=1,
        )

        curr_lr = get_lr_values(optimizer=optimizer)

        print("Finished Training")

        # Update SCAFFOLD terms
        scaffold_helper.terms_update(
            model=model,
            curr_lr=curr_lr,
            c_global_para=c_global_para,
            c_local_para=c_local_para,
            model_global=model_global,
        )

        # Evaluate local model
        _, test_acc = model.evaluate(x=test_ds, verbose=2)

        with tf_summary_writer.as_default():
            tf.summary.scalar("local_model_accuracy", test_acc, input_model.current_round)
        print(f"Accuracy of the model: {test_acc * 100:.2f}%")

        # Send model back to server with SCAFFOLD controls
        output_model = flare.FLModel(
            params={layer.name: layer.get_weights() for layer in model.layers},
            metrics={"accuracy": test_global_acc},
            meta={
                AlgorithmConstants.SCAFFOLD_CTRL_DIFF: scaffold_helper.get_delta_controls(),
            },
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()
