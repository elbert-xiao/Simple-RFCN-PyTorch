import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from utils.bbox_tools import generate_anchor_base
from utils.creator_tool import ProposalCreator


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """
    Enumerate all shifted anchors:

    :param anchor_base:     base anchor，shape: (A, 4), here 4==(y1, x1, y2, x2)
    :param feat_stride:     int, stride
    :param height:          height of RPN input feature map
    :param width:           width of RPN input feature map
    :return:                all anchor
    """
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # offset of center
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]  # the number of base anchor
    K = shift.shape[0]  # anchor group (==height * width)

    # A (base) anchor on each pixel <----> K offset，==>K * A anchors
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))  # shape:(K, A, 4)
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)  # shape:(K*A, 4)
    return anchor


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(self, in_channels=1024, mid_channels=512,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 feat_stride=16,
                 proposal_creator_params=dict()):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales,
                                                ratios=ratios)
        self.feat_stride = feat_stride

        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)

        # the number of base anchor
        n_anchor = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, (3, 3), 1, 1)

        # confidence and regression params
        score_out_channels = n_anchor * 2  # 2class(P/N) for each anchor
        self.score = nn.Conv2d(mid_channels, score_out_channels, 1)

        loc_out_channels = n_anchor * 4  # 4coords for each anchor
        self.loc = nn.Conv2d(mid_channels, loc_out_channels, 1)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1., only_rpn=False):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(self.anchor_base,
                                           self.feat_stride,
                                           hh, ww)
        n_anchor = self.anchor_base.shape[0]

        mid_out = F.relu(self.conv1(x))  # Dimension reduction+relu

        rpn_locs = self.loc(mid_out)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view((n, -1, 4))

        rpn_scores = self.score(mid_out)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()

        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        if only_rpn:
            # return reg and cls item of rpn
            return rpn_locs, rpn_scores, anchor

        rois_allbatch = list()
        rois_indices = list()
        for i in range(n):
            rois = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)  # shape:(S, 4)
            batch_index = i * np.ones((len(rois),), dtype=np.int32)  # shape: (S, )
            rois_allbatch.append(rois)  # [array[[], [], ...], array[[], [], ...] ]
            rois_indices.append(batch_index)  # roi batch index, [array[0, 0,...], array([1, 1,...], ...)]

        rois_allbatch = np.concatenate(rois_allbatch,
                                       axis=0)      # array([[y11, x11, y12, x12], [y21, x21, y22, x22], ...])
        rois_indices = np.concatenate(rois_indices, axis=0)  # array([0, 0,  ..., 1,1, 1, ...])

        return rpn_locs, rpn_scores, rois_allbatch, rois_indices, anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        if m.bias is not None:
            m.bias.data.zero_()
