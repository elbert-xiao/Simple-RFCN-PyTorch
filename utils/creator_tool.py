import numpy as np
import torch

from utils.bbox_tools import loc2bbox, bbox_iou, bbox2loc


class ProposalCreator:
    """
    generate proposal ROIs by call this class
    """
    def __init__(self,
                 rpn_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,  # on training mode: keep top-n1 bboxes before NMS
                 n_train_post_nms=2000,  # on training mode: keep top-n2 bboxes after NMS
                 n_test_pre_nms=6000,    # on test mode: keep top-n3 bboxes before NMS
                 n_test_post_nms=300,    # on test mode: keep top-n4 bboxes after NMS
                 min_size=16
                 ):
        self.rpn_model = rpn_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        """input should  be ndarray
        Propose RoIs.

        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Type of the output is same as the inputs.

        Args:
            loc (array): Predicted offsets and scaling to anchors.
                Its shape is :math:`(R, 4)`.
            score (array): Predicted foreground probability for anchors.
                Its shape is :math:`(R,)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(R, 4)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.

        Returns:
            array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.

        """
        # NOTE: when test, remember
        # r_fcn.eval()
        # to set self.traing = False
        if self.rpn_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert the anchors to the ROIs
        rois = loc2bbox(anchor, loc)

        # clip rois
        rois[:, slice(0, 4, 2)] = np.clip(
            rois[:, slice(0, 4, 2)], 0, img_size[0])
        rois[:, slice(1, 4, 2)] = np.clip(
            rois[:, slice(1, 4, 2)], 0, img_size[1])

        # remove small rois
        min_size = self.min_size * scale
        hs = rois[:, 2] - rois[:, 0]  # height
        ws = rois[:, 3] - rois[:, 1]  # width
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]

        rois = rois[keep, :]
        score = score[keep]

        # sorted by score
        # get topN anchors to NMS， e.g.N=12000(training)，6000(testing)
        order = score.ravel().argsort()[::-1]  # [::-1]表示倒序
        if n_pre_nms > 0:
            order = order[:n_pre_nms]  # shape:(n_pre_nms, )
        rois = rois[order, :]
        score = score[order]
        keep = torch.ops.torchvision.nms(
            torch.from_numpy(rois).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh
        )

        if n_post_nms > 0:
            keep = keep[:n_post_nms]

        rois = rois[keep.cpu().numpy()]
        # rois_score = score[keep.cpu().numpy()]

        return rois


class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals.

        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.

        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of
                the coordinates of bounding boxes.

        Returns:
            (array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        # get numbers of bbox
        n_bbox, _ = bbox.shape

        # Join GT bboxes
        roi = np.concatenate((roi, bbox), axis=0)
        # Preset number of positive samples
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)

        # get IOU between roi and bbox
        iou = bbox_iou(roi, bbox)
        # argmax index of each ROI
        gt_assignment = iou.argmax(axis=1)
        # max IOU of each ROI
        max_iou = iou.max(axis=1)

        # label of each ROI, the positive label start with 1
        gt_roi_label = label[gt_assignment] + 1

        # positive ROIs
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]

        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False
            )

        # negative ROIs
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        # the number of negative ROIs
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image

        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False
            )

        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0       # set the lable of neg ROIs to zero
        sample_roi = roi[keep_index]

        gt_roi_loc =  bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) /
                      np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator(object):
    """
    Assign the ground truth bounding boxes to anchors.

    params:
    n_sample            the numbers of sample anchors
    pos_iou_thresh      float, the anchor positive if its IOU with gt_bbox > pos_iou_thresh
    neg_iou_thresh      float, the anchor negative if its IOU with gt_bbox < neg_iou_thresh
    pos_ratio:          float, n_sample_pos / n_sample
    """
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio


    def __call__(self, gt_bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array):

            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """
        img_H, img_W = img_size
        n_anchor = len(anchor)

        # Get the index of anchors completely inside the image, e.g. array[0, 1, 3, 6]
        inside_index = _get_inside_index(anchor, img_H, img_W)

        anchor = anchor[inside_index]

        # shape: (inside_anchor_num, ) & (inside_anchor_num, )
        achor_argmax_ious, anchor_label = self._create_label(anchor, gt_bbox)

        # compute bounding box regression targets
        anchor_loc = bbox2loc(anchor, gt_bbox[achor_argmax_ious])  # shape:(inside_anchor_num, 4)

        # map up to original set of anchors
        anchor_label = _unmap(anchor_label, n_anchor, inside_index, fill=-1)  # shape:(n_anchor, )
        anchor_loc = _unmap(anchor_loc, n_anchor, inside_index, fill=0)       # shape:(n_anchor, 4)

        return anchor_loc, anchor_label

    def _create_label(self, anchor, gt_bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        anchor_label = np.empty((anchor.shape[0], ), dtype=np.int32)  # shape:(inside_anchor_num, 4)
        anchor_label.fill(-1)  # 初始化anchor标记为-1（弃用）

        anchor_argmax_ious, anchor_max_ious, gt_argmax_ious = self._calc_ious(anchor, gt_bbox)

        '''assign labels'''
        # assign negative labels first so that positive labels can clobber them
        anchor_label[anchor_max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        anchor_label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        anchor_label[anchor_max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(anchor_label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos),
                replace=False
            )
            anchor_label[disable_index] = -1  # reset to initial value (skip)

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(anchor_label == 1)
        neg_index = np.where(anchor_label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg),
                replace=False
            )
            anchor_label[disable_index] = -1

        return anchor_argmax_ious, anchor_label

    def _calc_ious(self, anchor, gt_bbox):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, gt_bbox)

        anchor_size, gt_bbox_size = ious.shape

        anchor_argmax_ious = ious.argmax(axis=1)
        anchor_max_ious = ious[np.arange(anchor_size), anchor_argmax_ious]

        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(gt_bbox_size)]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return anchor_argmax_ious, anchor_max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    """
    Unmap a subset of item (data) back to the original set of items (of size count)
    :param data:    current data，shape: ('N', 4) or ('N', )
    :param count:   the count of original data(dst data)
    :param index:   the index of the current data in the original data
    :param fill:    default value of raw data
    :return:
    """
    # one-dimensional data
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        # unmap high-dimensional data, i.e. shape:(count, 4)
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data

    return ret


def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]

    return index_inside



















