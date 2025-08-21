import numpy as np
import torch


class DepthMeter(object):
    def __init__(self, dataname):
        self.dataname = dataname
        self.total_rmses = 0.0
        self.abs_rel = 0.0
        self.n_valid = 0.0
        self.max_depth = 80.0
        self.min_depth = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()

        # Determine valid mask
        mask = torch.logical_and(gt < self.max_depth, gt > self.min_depth)
        self.n_valid += mask.float().sum().item()  # Valid pixels per image

        # Only positive depth values are possible
        # pred = torch.clamp(pred, min=1e-9)
        gt[gt <= 0] = 1e-9
        pred[pred <= 0] = 1e-9

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += rmse_tmp.sum().item()
        self.abs_rel += (torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item()

    def reset(self):
        self.total_rmses = 0.0
        self.abs_rel = 0.0
        self.n_valid = 0.0

    def get_score(self):
        if self.dataname == 'nyud':
            eval_result = {'RMSE': np.sqrt(self.total_rmses / self.n_valid)}
        else:
            raise NotImplementedError

        return eval_result
