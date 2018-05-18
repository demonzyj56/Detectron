"""Multibox classification weights op."""


class MultiBoxClsWeightOp(object):
    """Output weight matrix for positive and hard negative entries.
    Positive and hard negative entries are labeled as 1 and others as 0.
    """

    def __init__(self, neg_pos, batch_size):
        """Since SSD digs hard negatives from each image, we retrieve batch
        size here."""
        self._neg_pos = neg_pos
        self._batch_size = batch_size

    def forward(self, inputs, outputs):
        """
        Input:
            spatial_prob: 2-D tensor of (batch_size x num_priors, num_classes).
            Gives probability prediction at each location.
            ssd_labels_int32: 1-D tensor of (batch_size x num_priors, ).
            Gives gt labels at each location.

        Output:
            ssd_label_weights: 1-D tensor of (batch_size x num_priors).
            Positive and hard negative entries are 1, else 0.
        """
        # TODO(leoyolo): move gather operation to device side.
        cls_prob = inputs[0].data.copy()
        cls_prob = cls_prob[np.arange(cls_prob.shape[0]), inputs[1].data]
        pos = (inputs[1].data > 0)
        cls_prob[pos] = 1  # suppress positives
        cls_prob = cls_prob.reshape(self._batch_size, -1)
        # smaller indices are what we look for
        loss_idx = cls_prob.argsort(axis=1)
        idx_rank = loss_idx.argsort(axis=1)

        num_pos = pos.sum(dim=1, keepdims=True)
        num_neg = np.minimum(num_pos * self._neg_pos, cls_prob.shape[1])
        neg = (idx_rank < num_neg)
        weight = (pos | neg).astype(np.float32).reshape(-1)

        outputs[0].reshape(weight.shape)
        outputs[0].data[...] = weight
