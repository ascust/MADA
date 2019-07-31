import yaml
from easydict import EasyDict as edict

config = edict()

config.backbone = "resnet101"
config.cache_folder = "cache"

config.model_root = "../model_zoo"
config.data_root = "../data_root"
config.kvstore = "device"
config.tag = "default"
config.mean = [.485, .456, .406]
config.var = [.229, .224, .225]
config.num_class = 19
config.p1 = 0.6
config.p2 = 0.8

config.TRAIN = edict()
config.TRAIN.source_dataset = "gtav"
config.TRAIN.source_shorter_min = 720
config.TRAIN.source_shorter_max = 720
config.TRAIN.source_min_scale = 0.7
config.TRAIN.source_max_scale = 1.3
config.TRAIN.source_crop_size = [1280, 720]
config.TRAIN.source_random_flip = True
config.TRAIN.source_random_gaussian = False
config.TRAIN.target_dataset = "cityscapes"
config.TRAIN.target_shorter_min = 512
config.TRAIN.target_shorter_max = 512
config.TRAIN.target_min_scale = 0.7
config.TRAIN.target_max_scale = 1.3
config.TRAIN.target_crop_size = [1024, 512]
config.TRAIN.target_random_flip = True
config.TRAIN.target_random_gaussian = False
config.TRAIN.disp_freq = 10

config.TRAIN.batch_size = 1
config.TRAIN.iters = 250000
config.TRAIN.seg_lr = 0.00025
config.TRAIN.adv_lr = 0.0001
config.TRAIN.loss_lambda = 0.001
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 1e-4
config.TRAIN.save_freq = 10000
config.TRAIN.workers = 4



config.EVAL = edict()
config.EVAL.dataset = "cityscapes"
config.EVAL.shorter_min = 512
config.EVAL.shorter_max = 512
config.EVAL.multi_eval = False
config.EVAL.tar_folder = "images"
config.EVAL.output_folder = "outputs"
config.EVAL.workers = 4

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                raise ValueError("key %s must exist in config.py" % k)