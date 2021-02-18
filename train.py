import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time

from config import opt
from trainer import RFCN_Trainer
from data.dataset import TrainDataset, TestDataset, inverse_normalize
from utils.bbox_tools import toscalar, tonumpy, totensor
from rfcn_model import RFCN_ResNet101, RFCN
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rfcn_md_eval(dataloader, r_fcn: RFCN, test_num=10000):
    """
    eval model on the special dataset
    :param dataloader: test dataset
    :param r_fcn:      model
    :param test_num:
    :return:
    """
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    ii = -1
    for (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(dataloader):
        ii += 1
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        imgs = imgs.cuda()
        pred_bboxes_, pred_labels_, pred_scores_ = r_fcn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    return result


def RFCN_eval(load_path, **kwargs):
    """
    Ex:
    python train.py RFCN_eval --load_path='checkPoints/rfcn_voc07_0.725_ohem.pth' --test_num=5000

    :param test_num:
    :param kwargs:
    :return:
    """

    """parse params"""
    opt.parse(kwargs)
    opt.load_path = load_path

    """load train & test dataset"""
    print("load predict data")
    test_db = TestDataset()
    if opt.test_num < len(test_db):
        test_db = torch.utils.data.Subset(test_db, indices=torch.arange(opt.test_num))
    test_dataloader = DataLoader(test_db,
                                 shuffle=False,
                                 batch_size=opt.test_batch_size,
                                 num_workers=opt.test_num_workers,
                                 pin_memory=False)

    """create model"""
    rfcn_md = RFCN_ResNet101()
    print('model construct completed')

    rfcn_trainer = RFCN_Trainer(rfcn_md).cuda()
    if opt.load_path:
        rfcn_trainer.load(opt.load_path, load_viz_idx=opt.load_viz_x)
        print('load pretrained model from %s' % opt.load_path)

    eval_result = rfcn_md_eval(test_dataloader, rfcn_trainer.r_fcn, test_num=opt.test_num)
    print("map: ", str(eval_result['map']))


def RFCN_train(**kwargs):
    """
    python train.py RFCN_train
    """
    """parse params"""
    opt.parse(kwargs)
    opt.batch_size = 1   # force set batch_size to 1

    """load train & test dataset"""
    print('load data')
    train_db = TrainDataset()
    train_dataloader = DataLoader(train_db,
                                  shuffle=True,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  pin_memory=False)

    test_db = TestDataset()
    if opt.test_num < len(test_db):
        test_db = torch.utils.data.Subset(test_db, indices=torch.arange(opt.test_num))
    test_dataloader = DataLoader(test_db,
                                 shuffle=False,
                                 batch_size=opt.test_batch_size,
                                 num_workers=opt.test_num_workers,
                                 pin_memory=False)

    """create model"""
    rfcn_md = RFCN_ResNet101()
    print('model construct completed')

    rfcn_trainer = RFCN_Trainer(rfcn_md).cuda()
    if opt.load_path:
        rfcn_trainer.load(opt.load_path, load_viz_idx=opt.load_viz_x)
        print('load pretrained model parameters from %s' % opt.load_path)
        print("lr is:", rfcn_trainer.optimizer.param_groups[0]['lr'])
    rfcn_trainer.train()
    rfcn_trainer.viz.text(train_db.db.CLASS_NAME, win='labels')
    best_map = 0

    """training"""
    for epoch in range(opt.epoch_begin, opt.total_epoch):
        rfcn_trainer.reset_meters()
        step = -1
        for (img, bbox_, label_, scale) in tqdm(train_dataloader):
            step += 1
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            scale = scale.item()
            rfcn_trainer.train_step(img, bbox, label, scale)

            if (step + 1) % opt.print_interval_steps == 0:
                # plot loss
                for k, v in rfcn_trainer.get_meter_data().items():
                    rfcn_trainer.viz.line(Y=np.array([v]), X=np.array([rfcn_trainer.viz_index]),
                                         win=k,
                                         opts=dict(title=k, xlabel='px', ylable='loss'),
                                         update=None if rfcn_trainer.viz_index==0 else 'append')
                rfcn_trainer.viz_index += 1

                # plot ground truth bboxes
                ori_img_ = inverse_normalize(tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     tonumpy(bbox_[0]),
                                     tonumpy(label_[0]))
                rfcn_trainer.viz.image(gt_img, win='gt_img', opts={'title': 'gt_img'})

                # plot predict bboxes
                b_bboxes, b_labels, b_scores = rfcn_trainer.r_fcn.predict([ori_img_], visualize=True)

                pred_img = visdom_bbox(ori_img_,
                                       tonumpy(b_bboxes[0]),
                                       tonumpy(b_labels[0]).reshape(-1),
                                       tonumpy(b_scores[0]))
                rfcn_trainer.viz.image(pred_img, win='pred_img', opts={'title':'predict image'})

                # rpn confusion matrix(meter)
                rfcn_trainer.viz.text(str(rfcn_trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                rfcn_trainer.viz.image(rfcn_trainer.roi_cm.value().astype(np.uint8), win='roi_cm',
                                       opts={'title': 'roi_cm'})

        # get mAP
        eval_result = rfcn_md_eval(test_dataloader, rfcn_md, test_num=opt.test_num)
        lr_ = rfcn_trainer.optimizer.param_groups[0]['lr']
        log_info = 'epoch:{}, lr:{}, map:{},loss:{}'.format(str(epoch), str(lr_),
                                                  str(eval_result['map']),
                                                  str(rfcn_trainer.get_meter_data()))

        # plot mAP
        rfcn_trainer.viz.line(Y=np.array([eval_result['map']]), X=np.array([epoch]),
                              win='test_map',
                              opts=dict(title='test_map', xlabel='px', ylable='mAP'),
                              update=None if epoch == 0 else 'append')
        # plot log text
        rfcn_trainer.log(log_info)
        print(log_info)

        # if eval_result['map'].item() > best_map:
        cur_map = eval_result['map']
        cur_path = rfcn_trainer.save(best_map=cur_map)
        if cur_map > best_map:
            best_map = cur_map
            best_path = cur_path

        print("save model parameters to path: {}".format(cur_path))

        # update learning rate
        if (epoch + 1) in opt.LrMilestones:
            rfcn_trainer.load(best_path)
            print('update trainer weights from ', best_path, ' epoch is:', epoch)
        rfcn_trainer.scale_lr(epoch=epoch, gamma=opt.lr_gamma)

if __name__ == '__main__':
    import fire
    fire.Fire()

# python train.py RFCN_train --load_path='/home/elbert/projects/rfcn/checkPoints/rfcn_voc07_0.725_ohem.pth' \
#        --load_resnet101_path='/home/elbert/projects/rfcn/weights/resnet101-5d3b4d8f.pth' --rfcn_init_lr=0.0001 --total_epoch=8