import numpy as np
import torch


class SaliencyMeter(object):

    def __init__(self, ignore_index=255, threshold_step=0.05, beta=0.3):
        self.ignore_index = ignore_index
        self.beta = beta
        self.thresholds = torch.arange(threshold_step, 1, threshold_step)
        self.true_positives = torch.zeros(len(self.thresholds))
        self.predicted_positives = torch.zeros(len(self.thresholds))
        self.actual_positives = torch.zeros(len(self.thresholds))
        self.ious = []

    @torch.no_grad()
    def update(self, preds, targets):
        preds = preds.float() / 255.

        if targets.shape[1] == 1:
            targets = targets.squeeze(1)

        assert preds.shape == targets.shape

        for i in range(preds.size(0)):
            pred = preds[i]
            target = targets[i]
            valid_mask = (target != self.ignore_index)
            iou = np.zeros(len(self.thresholds))

            for idx, thresh in enumerate(self.thresholds):
                # threshold probablities
                f_pred = (pred >= thresh).long()
                f_target = target.long()

                f_pred = torch.masked_select(f_pred, valid_mask)
                f_target = torch.masked_select(f_target, valid_mask)

                self.true_positives[idx] += torch.sum(f_pred * f_target).item()
                self.predicted_positives[idx] += torch.sum(f_pred).item()
                self.actual_positives[idx] += torch.sum(f_target).item()

                iou[idx] = torch.sum(f_pred & f_target).item() / torch.sum(f_pred | f_target).item()

            self.ious.append(iou)

    def get_score(self):
        """
        Computes F-scores over state and returns the max.
        """
        precision = self.true_positives.float() / (self.predicted_positives + 1e-8)
        recall = self.true_positives.float() / (self.actual_positives + 1e-8)

        num = (1 + self.beta) * precision * recall
        denom = self.beta * precision + recall

        # For the rest we need to take care of instances where the denom can be 0
        # for some classes which will produce nans for that class
        fscore = num / (denom + 1e-8)
        fscore[fscore != fscore] = 0

        mIoUs = np.mean(np.array(self.ious), axis=0)

        eval_result = {'maxF': (fscore.max().item() * 100), 'mIoU': (mIoUs.max() * 100)}

        return eval_result
