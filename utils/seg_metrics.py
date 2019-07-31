import threading
import numpy as np
import mxnet as mx
from mxnet.metric import EvalMetric

__all__ = ['SegMetric']

class SegMetric(EvalMetric):
    """Computes pixAcc and mIoU metric scroes
    """
    def __init__(self, nclass, ignored_label=255):
        self.nclass = nclass
        self.ignored_label = ignored_label
        self.confMat = np.zeros(shape=(self.nclass, self.nclass),dtype=np.ulonglong)
        super(SegMetric, self).__init__('pixAcc & mIoU')

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NDArray' or list of `NDArray`
            The labels of the data.

        preds : 'NDArray' or list of `NDArray`
            Predicted values.
        """

        for (label, pred) in zip(labels, preds):
            self.eval_pair(label, pred)
            
    def eval_pair(self, label, pred):
        assert label.size == pred.size
        valid_index = label.flat != self.ignored_label
        gt = np.extract(valid_index, label.flat)
        p = np.extract(valid_index, pred.flat)
        temp = np.ravel_multi_index(np.array([gt.astype(np.uint8), p.astype(np.uint8)]), (self.confMat.shape))
        temp_mat = np.bincount(temp, minlength=np.prod(self.confMat.shape)).reshape(self.confMat.shape)
        self.confMat += temp_mat.astype(np.uint32)
    def get(self):
        pixAcc = float(np.diag(self.confMat).sum())/self.confMat.sum()
        ious = self._calc_ious()
        mIoU = ious.mean()
        return pixAcc, mIoU, ious

    def _calc_ious(self):
        scores = []
        for i in range(self.nclass):
            res = 0
            tp = np.ulonglong(self.confMat[i, i])
            gti = np.ulonglong(self.confMat[i, :].sum())
            resi = np.ulonglong(self.confMat[:, i].sum())
            denom = gti+resi-tp
            if denom != 0:
                res = float(tp)/denom
            scores.append(res)
        return np.array(scores)

    def reset(self):
        self.confMat[:] = 0
