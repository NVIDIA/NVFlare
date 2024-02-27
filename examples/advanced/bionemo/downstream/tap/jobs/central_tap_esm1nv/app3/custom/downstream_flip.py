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
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

# (0): import nvflare lightning api
import nvflare.client.lightning as flare


@hydra_runner(config_path=".", config_name="downstream_flip_tap")  # ESM
# @hydra_runner(config_path="../prott5nv/conf", config_name="downstream_flip_sec_str") # ProtT5
def main(cfg) -> None:
    logging.info("\n\n************* Finetune config ****************")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer = setup_trainer(cfg, builder=None)
        model = FineTuneProteinModel(cfg, trainer)
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

        # (1): flare patch
        flare.patch(trainer)

        # (2): Add while loop to keep receiving the FLModel in each FL round.
        # Note, after flare.patch the trainer.fit/validate will get the
        # global model internally at each round.
        while flare.is_running():
            # (optional): get the FL system info
            fl_sys_info = flare.system_info()
            print("--- fl_sys_info ---")
            print(fl_sys_info)

            # (3) evaluate the current global model to allow server-side model selection.
            print("--- validate global model ---")
            trainer.validate(model)

            # (4) Perform local training starting with the received global model.
            print("--- train new model ---")

            trainer.fit(model)
            logging.info("************** Finished Training ***********")

        if cfg.do_testing:
            logging.info("************** Starting Testing ***********")
            if "test" in cfg.model.data.dataset:
                trainer.test(model)
            else:
                raise UserWarning(
                    "Skipping testing, test dataset file was not provided. Please specify 'dataset.test' in yaml config"
                )
            logging.info("************** Finished Testing ***********")
    else:
        logging.info("************** Starting Preprocessing ***********")
        preprocessor = FLIPPreprocess()
        preprocessor.prepare_all_datasets(output_dir=cfg.model.data.preprocessed_data_path)


if __name__ == "__main__":
    main()
