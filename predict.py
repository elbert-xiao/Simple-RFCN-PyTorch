import os
import numpy as np
import cv2
from tqdm import tqdm

from utils.img_processing import read_image
from rfcn_model import RFCN_ResNet101
from trainer import RFCN_Trainer
from config import opt
from utils.vis_tool import visdom_bbox
from utils.bbox_tools import tonumpy


def predict(load_path, **kwargs):
    """
    """
    """parse parameters"""
    opt.load_path = load_path
    opt.parse(kwargs)

    """get images to be predicted"""
    if not os.path.isdir(opt.predict_output_dir):
        os.mkdir(opt.predict_output_dir)

    img_files = os.listdir(opt.predict_input_dir)
    img_files.sort()

    img_paths = [os.path.join(opt.predict_input_dir, name) for name in img_files]

    """create model"""
    rfcn_md = RFCN_ResNet101()
    print('model construct completed')

    rfcn_trainer = RFCN_Trainer(rfcn_md).cuda()
    if opt.load_path:
        rfcn_trainer.load(opt.load_path, load_viz_idx=opt.load_viz_x)
        print('load pretrained model from %s' % opt.load_path)

    """predict"""
    for img_path in tqdm(img_paths):
        raw_img = read_image(img_path, color=True)

        # plot predict bboxes
        b_bboxes, b_labels, b_scores = rfcn_trainer.r_fcn.predict([raw_img], visualize=True)
        pred_img = visdom_bbox(raw_img,
                               tonumpy(b_bboxes[0]),
                               tonumpy(b_labels[0]).reshape(-1),
                               tonumpy(b_scores[0]))

        file_name, file_ext = os.path.splitext(os.path.basename(img_path))
        result = np.hstack([b_labels[0][:, np.newaxis], b_scores[0][:, np.newaxis], b_bboxes[0]])

        # output to file
        file_out_path = os.path.join(opt.predict_output_dir, 'res_' + file_name+'.txt')
        np.savetxt(file_out_path, result, fmt='%.2f', delimiter=',')

        img_out_path = os.path.join(opt.predict_output_dir, file_name+'_res.jpg')
        pred_img = np.flipud(pred_img).transpose((1, 2, 0)) * 255
        cv2.imwrite(img_out_path, pred_img)

    print('Done!')


if __name__ == '__main__':
    import fire
    fire.Fire()



