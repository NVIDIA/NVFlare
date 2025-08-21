import numpy as np
import torch


class SemsegMeter(object):
    def __init__(self, dataname, ignore_index=255):
        if dataname == 'pascalcontext':
            n_classes = 20
            has_bg = True
        elif dataname == 'nyud':
            n_classes = 40
            has_bg = False
        else:
            raise NotImplementedError

        self.ignore_index = ignore_index
        self.n_classes = n_classes + int(has_bg)
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        valid = (gt != self.ignore_index)

        for i_part in range(self.n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

    def get_score(self):
        jac = [0] * self.n_classes
        for i_part in range(self.n_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = {'mIoU': (np.mean(jac) * 100)}

        return eval_result
