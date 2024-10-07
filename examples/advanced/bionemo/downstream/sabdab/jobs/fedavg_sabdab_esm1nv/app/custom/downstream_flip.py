# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from bionemo.data import FLIPPreprocess
from bionemo.data.metrics import accuracy, mse, per_token_accuracy
from bionemo.model.protein.downstream import FineTuneProteinModel
from bionemo.model.utils import setup_trainer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf, open_dict

# Import nvflare lightning API for federated learning
import nvflare.client.lightning as flare
from nvflare.client.api import init

micro_batch_size = 32
val_check_intervals = {
    "site-1": min(int(416 / micro_batch_size), 3),  # Use min to ensure it's <= 3
    "site-2": min(int(238 / micro_batch_size), 3),
    "site-3": min(int(282 / micro_batch_size), 3),
    "site-4": min(int(472 / micro_batch_size), 3),
    "site-5": min(int(361 / micro_batch_size), 3),
    "site-6": min(int(157 / micro_batch_size), 3),
}


@hydra_runner(config_path=".", config_name="downstream_flip_sabdab")  # ESM1
def main(cfg) -> None:
    logging.info("\n\n************* Finetune config ****************")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    init()
    # Get FL system info and set site-specific parameters
    fl_sys_info = flare.system_info()
    site_name = fl_sys_info["site_name"]
    cfg.model.data.dataset.train = f"sabdab_chen_{site_name}_train"
    cfg.trainer.val_check_interval = val_check_intervals[site_name]
    print(f"Running client {site_name} with train data: {cfg.model.data.dataset.train}")
    print(f"Validation check interval: {cfg.trainer.val_check_interval}")

    # Do preprocessing if specified in config
    if cfg.do_preprocessing:
        logging.info("************** Starting Preprocessing ***********")
        preprocessor = FLIPPreprocess()
        preprocessor.prepare_all_datasets(output_dir=cfg.model.data.preprocessed_data_path)

    if not cfg.do_training and not cfg.do_testing:
        return

    trainer = setup_trainer(cfg, builder=None, reset_accumulate_grad_batches=False)

    # Load model
    with open_dict(cfg):
        cfg.model.encoder_cfg = cfg

    if cfg.restore_from_path:
        logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
        model = FineTuneProteinModel.restore_from(
            cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
        )
    else:
        model = FineTuneProteinModel(cfg.model, trainer)

    metrics = {}
    metrics_args = {}
    for idx, name in enumerate(cfg.model.data.target_column):
        if cfg.model.data.task_type == "token-level-classification":
            metrics[name + "_accuracy"] = per_token_accuracy
            metrics_args[name + "_accuracy"] = {"label_id": idx}
        elif cfg.model.data.task_type == "classification":
            metrics[name + "_accuracy"] = accuracy
            metrics_args[name + "_accuracy"] = {}
        elif cfg.model.data.task_type == "regression":
            metrics[name + "_MSE"] = mse
            metrics_args[name + "_MSE"] = {}

    model.add_metrics(metrics=metrics, metrics_args=metrics_args)

    # Patch trainer for NVFlare federated learning
    flare.patch(trainer)

    # Federated learning loop
    while flare.is_running():
        fl_sys_info = flare.system_info()
        print("--- fl_sys_info ---")
        print(fl_sys_info)

        # Validate current global model
        print("--- validate global model ---")
        trainer.validate(model)

        # Perform local training with received global model
        print("--- train new model ---")
        trainer.fit(model)
        logging.info("************** Finished Training ***********")

    if cfg.do_testing:
        logging.info("************** Starting Testing ***********")
        if "test" in cfg.model.data.dataset:
            trainer.limit_train_batches = 0
            trainer.limit_val_batches = 0
            trainer.fit(model)
            trainer.test(model, ckpt_path=None)
        else:
            raise UserWarning(
                "Skipping testing, test dataset file was not provided. Please specify 'dataset.test' in yaml config"
            )
        logging.info("************** Finished Testing ***********")


if __name__ == "__main__":
    main()
