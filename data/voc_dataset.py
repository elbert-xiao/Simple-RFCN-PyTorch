import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import os
import numpy as np
from utils.img_processing import read_image


class VOCDataset(Dataset):
    CLASS_NAME = (
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor')

    def __init__(self, root_dir,
                 split='trainval',
                 use_difficult=False,
                 return_difficult=False):
        self.root_dir = root_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult

        id_list_file = os.path.join(
            root_dir, 'ImageSets/Main/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        t_labels, t_bboxes, t_difficult = list(), list(), list()
        file_id = self.ids[i]
        tree = ET.parse(
            os.path.join(self.root_dir, 'Annotations', file_id + '.xml'))

        for obj in tree.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            t_labels.append(self.CLASS_NAME.index(obj.find('name').text.lower().strip()))
            t_difficult.append(int(obj.find('difficult').text))

            box = obj.find('bndbox')
            y_min = int(box.find('ymin').text) - 1  # subtract 1 to make pixel indexes 0-based
            x_min = int(box.find('xmin').text) - 1
            y_max = int(box.find('ymax').text) - 1
            x_max = int(box.find('xmax').text) - 1
            t_bboxes.append([y_min, x_min, y_max, x_max])

        t_labels = np.stack(t_labels).astype(np.int32)
        t_bboxes = np.stack(t_bboxes).astype(np.float32)

        # When `use_difficult==False`, all elements in `difficult` are False.
        t_difficult = np.array(t_difficult, dtype=np.bool).astype(np.uint8)

        # Load a image
        img_file = os.path.join(self.root_dir, 'JPEGImages', file_id + '.jpg')
        img = read_image(img_file, color=True)

        return img, t_bboxes, t_labels, t_difficult

    __getitem__ = get_example


if __name__ == '__main__':
    '''===== voc dataset test ====='''
    from config import opt
    import matplotlib.pyplot as plt

    voc07_dataset = VOCDataset(opt.voc07_data_dir)
    img, bbox, label, difficult = voc07_dataset.get_example(0)
    print("bbox:\n", bbox)
    print("label:\n", label)
    print("difficult:\n", difficult)

    plt.imshow(img.transpose((1, 2, 0)).astype(np.int))
    plt.show()