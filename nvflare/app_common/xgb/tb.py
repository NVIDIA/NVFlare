import os

import xgboost as xgb
from xgboost import callback


class TensorBoardCallback(xgb.callback.TrainingCallback):
    def __init__(self, app_dir: str, tensorboard):
        self.train_writer = tensorboard.SummaryWriter(log_dir=os.path.join(app_dir, "train-auc/"))
        self.val_writer = tensorboard.SummaryWriter(log_dir=os.path.join(app_dir, "val-auc/"))

    def after_iteration(self, model, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog):
        if not evals_log:
            return False

        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                if data == "train":
                    self.train_writer.add_scalar(metric_name, score, epoch)
                else:
                    self.val_writer.add_scalar(metric_name, score, epoch)
        return False
