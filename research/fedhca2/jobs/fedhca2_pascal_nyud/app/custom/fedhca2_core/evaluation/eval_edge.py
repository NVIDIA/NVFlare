import torch

from losses import BalancedBCELoss


class EdgeMeter(object):

    def __init__(self, dataname, ignore_index=255):
        if dataname == 'pascalcontext':
            pos_weight = 0.95
        elif dataname == 'nyud':
            pos_weight = 0.95
        else:
            raise NotImplementedError

        self.loss = 0
        self.n = 0
        self.loss_function = BalancedBCELoss(pos_weight=pos_weight, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        valid_mask = (gt != self.ignore_index)
        pred = pred[valid_mask]
        gt = gt[valid_mask]

        pred = pred.float().squeeze() / 255.
        loss = self.loss_function(pred, gt).item()
        numel = gt.numel()
        self.n += numel
        self.loss += numel * loss

    def reset(self):
        self.loss = 0
        self.n = 0

    def get_score(self):
        eval_dict = {'loss': (self.loss / self.n)}

        return eval_dict
