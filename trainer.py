import torch
import torchvision
import torch.nn as nn
from collections import namedtuple
from torch.nn import functional as F
import visdom
from torchnet.meter import ConfusionMeter, AverageValueMeter
import time
import numpy as np
import os

from config import opt
from utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from utils.bbox_tools import tonumpy, totensor, toscalar
from rfcn_model import RFCN

RPN_LossTuple = namedtuple('RPN_LossTuple',
                           [
                               'rpn_loc_loss',
                               'rpn_cls_loss',
                               'total_loss'
                           ])

RFCN_LossTuple = namedtuple('RFCN_LossTuple',
                            [
                                'rpn_loc_loss',
                                'rpn_cls_loss',
                                'roi_loc_loss',
                                'roi_cls_loss',
                                'total_loss'
                            ])


class RFCN_Trainer(nn.Module):
    """
    trainer for RFCN, return loss:
    rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss

    params: r_fcn  --RFCN model
    """

    def __init__(self, r_fcn: RFCN):
        super(RFCN_Trainer, self).__init__()

        self.r_fcn = r_fcn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # generate anchor for RPN training
        self.anchor_target_creator = AnchorTargetCreator()

        proposal_target_num = 300 if opt.use_OHEM else 128
        self.proposal_target_creator = ProposalTargetCreator(n_sample=proposal_target_num)

        self.loc_normalize_mean = r_fcn.loc_normalize_mean
        self.loc_normalize_std = r_fcn.loc_normalize_std

        self.optimizer = self.get_optimizer()

        # visdom wrapper
        self.viz = visdom.Visdom(env=opt.viz_env)
        self.viz_index = 0
        self.log_text = ''

        # record training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(self.r_fcn.class_num)
        if opt.FIX_HEAD:
            self.meters = {k: AverageValueMeter() for k in RPN_LossTuple._fields}
        else:
            self.meters = {k: AverageValueMeter() for k in RFCN_LossTuple._fields}

    def forward(self, imgs, bboxes, labels, scale):
        """
        :param imgs:        variable with a batch of images.
        :param bboxes:      A batch of GT bounding boxes.
        :param labels:      labels of gt bboxes.
            Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
        :param scale:       Amount of scaling applied to the raw image during
                            preprocessing.
        :return:
            namedtuple of losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.r_fcn.extractor_phase1(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.r_fcn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]  # shape: (gt_num,)
        rpn_score = rpn_scores[0]  # shape: (anchor_num, 2)
        rpn_loc = rpn_locs[0]  # shape: (anchor_num, 4)
        roi = rois[np.where(roi_indices == 0)[0]]  # shape(R, 4)

        # --------------- rpn losses ------------ #
        anchor_loc_gt, anchor_label_gt = self.anchor_target_creator(
            tonumpy(bbox),
            anchor,
            img_size)

        anchor_loc_gt = totensor(anchor_loc_gt)
        anchor_label_gt = totensor(anchor_label_gt).long()
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            anchor_loc_gt,
            anchor_label_gt.data,
            self.rpn_sigma)

        rpn_cls_loss = F.cross_entropy(rpn_score, anchor_label_gt.cuda(), ignore_index=-1)

        with torch.no_grad():
            _anchor_label_gt = anchor_label_gt[anchor_label_gt > -1]
            _rpn_score = rpn_score[anchor_label_gt > -1]
            self.rpn_cm.add(_rpn_score, _anchor_label_gt.data.long())

        # **************** for head ****************
        if opt.FIX_HEAD:
            losses = [rpn_loc_loss, rpn_cls_loss]
            losses = losses + [sum(losses)]

            return RPN_LossTuple(*losses)
        else:
            # sample rois for Head training
            sample_roi, roi_loc_gt, roi_label_gt = self.proposal_target_creator(
                roi,
                tonumpy(bbox),
                tonumpy(label),
                self.loc_normalize_mean,
                self.loc_normalize_std)

            # Note: set all value to zero(batch_size == 1)
            sample_roi_index = torch.zeros(len(sample_roi), dtype=torch.float).cuda()
            sample_roi = totensor(sample_roi).float()

            roi_locs, roi_scores = self.r_fcn.head(
                features,
                sample_roi,
                sample_roi_index)

            # ----------- PsROI losses ----------- #
            roi_label_gt = totensor(roi_label_gt).long()
            roi_loc_gt = totensor(roi_loc_gt)

            n_sample = roi_locs.shape[0]
            roi_locs = roi_locs.view(n_sample, -1, 4)
            if opt.cls_reg_specific:
                roi_locs = roi_locs[torch.arange(n_sample), roi_label_gt]
            else:
                roi_locs = roi_locs[torch.arange(n_sample), 1]

            roi_loc_loss = _fast_rcnn_loc_loss(
                roi_locs.contiguous(),
                roi_loc_gt,
                roi_label_gt.data,
                self.roi_sigma,
                ohem=opt.use_OHEM)

            if opt.use_OHEM:
                roi_cls_loss = F.cross_entropy(roi_scores, roi_label_gt.cuda(), reduction='none')
                roi_cls_loss, roi_loc_loss = self.ohem_dectect_loss(roi_cls_loss, roi_loc_loss,
                                                                    roi_label_gt, sample_roi,
                                                                    use_nms=True,
                                                                    hard_num=opt.hard_num)
            else:
                roi_cls_loss = F.cross_entropy(roi_scores, roi_label_gt.cuda())

            with torch.no_grad():
                self.roi_cm.add(roi_scores, roi_label_gt.data.long())

            losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
            losses = losses + [sum(losses)]

            return RFCN_LossTuple(*losses)

    def ohem_dectect_loss(self, cls_loss, loc_loss, gt_label, rois, hard_num=128,
                          use_nms=True, nms_thresh=0.7):
        """
        :param cls_loss:       cls loss
        :param loc_loss:       reg loss
        :param gt_label:       gt label of rois
        :param rois:           sampled rois by proposalTarget module
        :param hard_num:       the number of rois for backward
        :param use_nms:        filter ROI with excessive overlap
        :param nms_thresh:     nms阈值
        :return:
        """
        bbox_loss = cls_loss + loc_loss

        if use_nms:
            # nms based on loss
            keep = torchvision.ops.nms(rois, bbox_loss, iou_threshold=nms_thresh)
            bbox_loss = bbox_loss[keep]
            cls_loss = cls_loss[keep]
            loc_loss = loc_loss[keep]
            gt_label = gt_label[keep]

        # the number of rois for backward
        back_size = min(len(bbox_loss), hard_num)

        # hard example mining
        if back_size < len(bbox_loss):
            _, top_idx = torch.topk(bbox_loss, back_size)
            top_cls_loss = cls_loss[top_idx]
            isFg = (gt_label > 0)[top_idx]
            top_fg_idx = top_idx[isFg]
            top_loc_loss = loc_loss[top_fg_idx] if len(top_fg_idx) > 0 else torch.tensor([0.]).cuda()
        else:
            top_cls_loss = cls_loss
            top_loc_loss = loc_loss

        top_cls_loss_normal = top_cls_loss.mean()
        top_loc_loss_normal = top_loc_loss.mean()

        return top_cls_loss_normal, top_loc_loss_normal

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.update_meters(losses)
        self.optimizer.step()

        return losses

    def update_meters(self, losses):
        loss_d = {k: toscalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

    def save(self, save_optimizer=False, save_path=None, best_map=0., **kwargs):
        save_dict = dict()

        save_dict['model'] = self.r_fcn.state_dict()
        save_dict['config'] = opt.state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = {'viz_index': self.viz_index}
        save_dict['best_map'] = best_map

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            save_path = './checkPoints/rfcn_'
            if opt.head_ver is not None:
                save_path += 'vgg_roi_'
            time_str = time.strftime('%m%d%H%M')
            save_path += '{}_{}.pth'.format(time_str, best_map)

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True,
             load_viz_idx=False,
             parse_opt=False):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.r_fcn.load_state_dict(state_dict['model'])
        else:
            raise ValueError("Cannot find the model parameters of RFCN, load_path:\n",
                             path)

        if load_viz_idx:
            self.viz_index = state_dict['vis_info']['viz_index']

        if parse_opt:
            print("Load history configuration!")
            opt.parse(state_dict['config'])

        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self


    def scale_lr(self, epoch, gamma=0.1):
        if (epoch + 1) in opt.LrMilestones:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= gamma
        return self.optimizer

    def get_optimizer(self):
        """
        return optimizer
        """
        lr = opt.rfcn_init_lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]

        return torch.optim.SGD(params=params, momentum=0.9)

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.viz.text(self.log_text, win, opts={"title": 'log_text'})


# multi-mask loss
def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma, ohem=False):
    if ohem:
        pred_loc_train = pred_loc[gt_label > 0]
        gt_loc_train = gt_loc[gt_label > 0]
        fg_loc_loss = F.smooth_l1_loss(pred_loc_train, gt_loc_train, reduction='none', beta=(1. / (sigma ** 2)))

        loc_loss = torch.zeros((len(gt_label), ), dtype=torch.float32).cuda()
        loc_loss[gt_label > 0] = torch.sum(fg_loc_loss, dim=1) / 4
    else:
        in_weight = torch.zeros(gt_loc.shape).cuda()

        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
        pred_loc_train = pred_loc[in_weight == 1]
        gt_loc_train = gt_loc[in_weight == 1]

        loc_loss = F.smooth_l1_loss(pred_loc_train, gt_loc_train, reduction='sum', beta=(1. / (sigma ** 2)))

        if (gt_label > 0).sum() == 0:
            loc_loss = 0.
        else:
            loc_loss /= (4 * (gt_label > 0).sum().float())

    return loc_loss

