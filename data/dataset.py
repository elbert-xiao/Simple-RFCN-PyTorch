from torch.utils.data import Dataset
import torch
from torchvision import transforms as tvtsf
from skimage import transform as sktsf

from .voc_dataset import VOCDataset
from config import opt
from utils.img_processing import resize_bbox, random_flip, flip_bbox


def inverse_normalize(img):
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalize(img):
    """
    normalize, return -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)

    # both the longer and shorter should be less than
    # max_size and min_size
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H*scale, W*scale), mode='reflect',
                       anti_aliasing=False)
    normalize = pytorch_normalize
    return normalize(img), scale


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img, scale = preprocess(img, self.min_size, self.max_size)

        _, o_H, o_W = img.shape
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = random_flip(img, x_random=True, return_param=True)
        bbox = flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class TrainDataset(Dataset):
    def __init__(self):
        self.db = VOCDataset(opt.voc07_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))

        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset(Dataset):
    def __init__(self, split='test', use_difficult=True):
        self.db = VOCDataset(opt.voc07_data_dir,
                             split=split,
                             use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img, _ = preprocess(ori_img)

        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
