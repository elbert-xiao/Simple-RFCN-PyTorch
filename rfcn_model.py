import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torchvision.ops import PSRoIPool

from torch_resnet import ResNet_BaseNet
from config import opt
from RPN import RegionProposalNetwork, normal_init
from utils.bbox_tools import totensor, tonumpy, loc2bbox
from data.dataset import preprocess
from utils.psroi_module import PSRoIPooling2D

class RFCN(nn.Module):
    """
    R-FCN base class
    """
    def __init__(self,
                 extractor_phase1,
                 rpn: RegionProposalNetwork,
                 head,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super(RFCN, self).__init__()
        self.extractor_phase1 = extractor_phase1
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        self.use_preset('evaluate')
        self.class_num = opt.class_num

        self.n_cls_reg = self.class_num if opt.cls_reg_specific else 2

    def forward(self, x, scale=1.):
        """
        :param x:       x (autograd.Variable): 4D image variable.
        :param scale:   Amount of scaling applied to the raw image during preprocessing.
        :return:
            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois_batch**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **rois_batch_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.
        """
        # height and width
        img_size = x.shape[2:]

        h = self.extractor_phase1(x)
        rpn_locs, rpn_scores, rois_batch, rois_batch_indices, anchor = \
            self.rpn(h, img_size, scale)

        # shape:(R, 4 * num_cls_reg) & (R, class_num_withBg)
        rois_batch = totensor(rois_batch).float()
        rois_batch_indices = totensor(rois_batch_indices)
        roi_locs, roi_scores = self.head(h, rois_batch, rois_batch_indices)

        return roi_locs, roi_scores, rois_batch, rois_batch_indices

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == "visualize":
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == "evaluate":
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError("preset must be 'visualize' or 'evaluate'")

    @torch.no_grad()
    def predict(self, imgs, sizes=None, visualize=False):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]  # original height & width
                img, _ = preprocess(tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            self.use_preset('evaluate')
            prepared_imgs = imgs

        b_bboxes = list()
        b_labels = list()
        b_scores = list()
        for i, (img, size) in enumerate(zip(prepared_imgs, sizes)):
            img = totensor(img[None]).float()  # Expand a dimension
            scale = img.shape[3] / size[1]     # scale ratio
            roi_locs, roi_scores, rois, roi_indices = self(img, scale=scale)

            # We are assuming that batch size is 1.
            roi_score = roi_scores.data    # shape: (Ri, self.class_num)
            roi_loc = roi_locs.data        # shape: (Ri, n_cls_reg * 4)

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            roi = totensor(rois) / scale

            # denormalize
            mean = torch.Tensor(self.loc_normalize_mean).cuda().repeat(self.n_cls_reg)[None]
            std = torch.Tensor(self.loc_normalize_std).cuda().repeat(self.n_cls_reg)[None]

            roi_loc = (roi_loc * std + mean)
            roi_loc = roi_loc.view(-1, self.n_cls_reg, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_loc)

            roi_bbox = loc2bbox(tonumpy(roi).reshape((-1, 4)),
                                tonumpy(roi_loc).reshape(-1, 4))
            roi_bbox = totensor(roi_bbox)
            roi_bbox = roi_bbox.view(-1, self.n_cls_reg * 4)

            # clip bounding box
            roi_bbox[:, 0::2] = (roi_bbox[:, 0::2]).clamp(min=0, max=size[0])
            roi_bbox[:, 1::2] = (roi_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = F.softmax(totensor(roi_score), dim=1)

            bboxes, labels, scores = self._suppress(roi_bbox, prob)
            b_bboxes.append(bboxes)
            b_labels.append(labels)
            b_scores.append(scores)

        self.train()
        return b_bboxes, b_labels, b_scores

    def _suppress(self, raw_bbox, raw_prob):
        """
        NMS for each class
        :param raw_cls_bbox:    Tensor, all predict bboxes
        :param raw_prob:        Tensor, confidence of predict bboxes after softmax
        :return:
        """
        raw_bbox = raw_bbox.reshape((-1, self.n_cls_reg, 4))

        bboxes, labels, scores = list(), list(), list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.class_num):
            # class agnostic: the same regression factors for all classes(different conf)
            # class specific: different regression factors for different class
            if opt.cls_reg_specific:
                bbox_l = raw_bbox[:, l, :]
            else:
                bbox_l = raw_bbox[:, 1, :]
            prob_l = raw_prob[:, l]

            # filter by confidence threshold
            mask = prob_l > self.score_thresh
            bbox_l = bbox_l[mask]
            prob_l = prob_l[mask]

            keep = torchvision.ops.nms(bbox_l, prob_l, self.nms_thresh)

            bboxes.append(bbox_l[keep])
            scores.append(prob_l[keep])
            # predict label is 0-19
            labels.append((l-1) * np.ones((len(keep), ), dtype=np.int32))

        bboxes = tonumpy(torch.cat(bboxes, dim=0)).astype(np.float32)
        scores = tonumpy(torch.cat(scores, dim=0)).astype(np.float32)
        labels = np.concatenate(labels, axis=0).astype(np.int32)

        return bboxes, labels, scores


class ResNet101_PsROI_Head(nn.Module):
    """
    ROI Head of R-FCN
    """
    def __init__(self, class_num, k, spatial_scale, extractor_phase2):
        """
        :param class_num: the number of classes (include background)
        :param k:         the number of bin for psRoI
        :param spatial_scale     stride of the feature extractor（ex:1/16.）
        :param extractor_phase2  feature extractor 2
        """
        super(ResNet101_PsROI_Head, self).__init__()
        self.class_num = class_num
        self.k = k
        self.spatial_scale = spatial_scale
        self.n_cls_reg = class_num if opt.cls_reg_specific else 2

        self.extractor_phase2 = extractor_phase2
        self.generatePsScoreMap = nn.Conv2d(1024, self.k * self.k * self.class_num, kernel_size=(1, 1), bias=True)
        self.generateLocMap = nn.Conv2d(1024, self.k * self.k * self.n_cls_reg * 4, kernel_size=(1, 1), bias=True)

        # self.psROI_score = PSRoIPool(self.k, spatial_scale=self.spatial_scale)
        # self.psROI_loc = PSRoIPool(self.k, spatial_scale=self.spatial_scale)
        self.psROI_score = PSRoIPooling2D(pool_size=self.k, spatial_scale=self.spatial_scale)
        self.psROI_loc = PSRoIPooling2D(pool_size=self.k, spatial_scale=self.spatial_scale)

        self.avg_pool_score = nn.AvgPool2d(kernel_size=self.k, stride=self.k)
        self.avg_pool_loc = nn.AvgPool2d(kernel_size=self.k, stride=self.k)

        from utils.psroi_module import acitvate_PsROI_for_eval
        acitvate_PsROI_for_eval(self.psROI_score)

        normal_init(self.generatePsScoreMap, 0, 0.01)
        normal_init(self.generateLocMap, 0, 0.01)

    def forward(self, x, rois, roi_indices):
        """
        forward of psRoI
        :param x:               input feature map
        :param rois:            rois, torch.tensor, shape:(S1+...+Sn, 4), here 4<==>(y_min, x_min, y_max, x_max)
        :param roi_indices:     Batch to which it belongs, shape: torch.tensor([0, 0, ..., 1, 1, ...])
        :return:
                roi_locs, (tx, ty, tw, th), shape:(sum(roi_num_i), 4)
                roi_scores, class confidence， shape:(sum(roi_num_i), ClassNum)
        """
        """combine rois and indices"""
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)

        # ([y_min, x_min, y_max, x_max] ==> [x_min, y_min, x_max, y_max])
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        """extract feature again"""
        h = self.extractor_phase2(x)

        """roi classification"""
        score_map = self.generatePsScoreMap(h)  # channels: k^2 * (C+1), shape:(b, C, H, W)

        # shape:(sum(roi_num_i) for all batch, ClassNum, k, k)
        score_pooling = self.psROI_score(score_map, indices_and_rois)
        roi_scores = self.avg_pool_score(score_pooling)   # shape:(sum(roi_num_i), ClassNum, 1, 1)
        roi_scores = roi_scores.squeeze()        # shape:(sum(roi_num_i), ClassNum)

        """roi regression"""
        loc_map = self.generateLocMap(h)              # channels: k^2 * 4 * n_cls_reg, shape:(b, C, H, W)
        # shape:(sum(roi_num_i) for all batch, n_cls_reg * 4, k, k)
        loc_pooling = self.psROI_loc(loc_map, indices_and_rois)
        roi_locs = self.avg_pool_loc(loc_pooling)    # shape:(sum(roi_num_i), n_cls_reg * 4, 1, 1)

        roi_locs = roi_locs.squeeze()       # shape:(sum(roi_num_i), n_cls_reg * 4) ==> here 4 is (ty, tx, tw, th)

        return roi_locs, roi_scores


from torchvision.ops import RoIPool
class VGG16RoIHead(nn.Module):
    """
    ROI Head of Faster R-CNN
    """
    def __init__(self, n_class, roi_size, spatial_scale, extractor_phase2):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.extractor_phase2 = extractor_phase2
        from torchvision.models import vgg16

        vgg_weights_path = '/home/elbert/mount/win_data/model_para/vgg16-dict.pth'
        model = vgg16(pretrained=False)
        model.load_state_dict(torch.load(vgg_weights_path))
        classifier = model.classifier
        del classifier[6]
        del classifier[5]
        del classifier[2]

        self.n_cls_reg = n_class if opt.cls_reg_specific else 2

        self.classifier = classifier

        self.cls_loc = nn.Linear(4096, self.n_cls_reg * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)

        # (yx ==> xy)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()
        h = self.extractor_phase2(x)

        pool = self.roi(h, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        return roi_cls_locs, roi_scores


class RFCN_ResNet101(RFCN):
    """
    R-FCN base on resnet101
    """

    # stride ==> 16
    feat_stride = 16

    def __init__(self,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]):
        extractor_phase1, extractor_phase2 = ResNet_BaseNet(opt.load_resnet101_path)

        # fix ResNet parameters
        for layer in extractor_phase1[:4+opt.FIXED_BLOCKS]:
            for p in layer.parameters():
                p.requires_grad = False

        normal_init(extractor_phase2._modules['dim_sub'], 0, 0.01)

        # fix BN layer
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        extractor_phase1.apply(set_bn_fix)
        extractor_phase2.apply(set_bn_fix)

        rpn = RegionProposalNetwork(
            in_channels=1024,
            mid_channels=512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride)

        # fix RPN parameters
        if opt.FIX_RPN:
            for p in rpn.parameters():
                p.requires_grad = False

        head = ResNet101_PsROI_Head(opt.class_num, opt.roi_k,
                                    spatial_scale=(1. / self.feat_stride),
                                    extractor_phase2=extractor_phase2)
        if opt.head_ver is not None:
            head = VGG16RoIHead(opt.class_num, opt.roi_k, spatial_scale=(1. / self.feat_stride),
                                extractor_phase2=extractor_phase2)  # vgg16 roi head

        # fix Head parameters
        if opt.FIX_HEAD:
            for p in extractor_phase2.parameters():
                p.requires_grad = False
            for p in head.parameters():
                p.requires_grad = False

        super(RFCN_ResNet101, self).__init__(
            extractor_phase1,
            rpn,
            head)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)

        if mode:
            self.extractor_phase1.eval()
            for fix_layer in range(6, 3 + opt.FIXED_BLOCKS, -1):
                self.extractor_phase1[fix_layer].train()

            # Set batchnorm always in eval mode during training or testing!
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.extractor_phase1.apply(set_bn_eval)
            self.head.extractor_phase2.apply(set_bn_eval)


