from pprint import pprint


class Config:

    # Dataset path
    voc07_data_dir = '/home/elbert/datasets/VOC2007'
    class_num = (1 + 20)   # num of classes

    min_size = 600
    max_size = 1000

    num_workers = 8
    test_num_workers = 8

    print_interval_steps = 500

    batch_size = 1
    test_batch_size = 1

    total_epoch = 16
    epoch_begin = 0

    # RFCN lr setting
    weight_decay = 0.0005
    rfcn_init_lr = 0.001
    lr_gamma = 0.1   # decay
    LrMilestones = [10, ]

    # fix parameters of layer of resnet /RPN /psRoI head
    FIXED_BLOCKS = 1
    FIX_RPN = False
    FIX_HEAD = False

    use_OHEM = True  # whether to use roi OHEM loss
    hard_num = 128

    # the path of checkpoints
    save_dir = None
    load_path = None
    load_viz_x = False   # read the 'Visdom' index of last training process

    # the path of resnet101 weights
    load_resnet101_path="./weights/resnet101-5d3b4d8f.pth"

    roi_k = 7

    # sigma for smooth L1 loss
    rpn_sigma = 3.  # Abscissa of the intersection of L1 and L2 loss curve is: (1. / sigma**2)
    roi_sigma = 1.

    # Visdom
    viz_env = 'R-FCN'

    test_num = 5000

    # head
    head_ver = None
    cls_reg_specific = False

    # predict
    predict_input_dir = 'predict/imgs'
    predict_output_dir = 'predict/results'

    def parse(self, kwargs):
        state_dict = self.state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            print(k, ":", v)
            setattr(self, k, v)

        print('======user config========')
        pprint(self.state_dict())
        print('==========end============')

    def state_dict(self):
        return {k:getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()