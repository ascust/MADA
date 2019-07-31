import os
from tqdm import tqdm
import numpy as np
import argparse

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

import gluoncv
from datasets import get_segmentation_dataset
from gluoncv.data import ms_batchify_fn
from gluoncv.utils.viz import get_color_pallete
from networks.segbase import *
from utils.seg_metrics import SegMetric

from utils import config

import logging
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cfg', type=str, default=None,
                        help='config file')
    parser.add_argument('--gpu', type=int,
                        default=0,
                        help='gpu index')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')

    parser.add_argument('--saveres', action="store_true",
                        help='save the result')
    # the parser
    args = parser.parse_args()

    args.ctx = mx.gpu(args.gpu)
    return args

def test(config):
    if not os.path.exists(config.resume[:-7]):
        os.mkdir(config.resume[:-7])
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.var),
    ])
    # dataset and dataloader

    testset = get_segmentation_dataset(
        config.EVAL.dataset, root=config.data_root, split='val', mode='testval', transform=input_transform, num_class=config.num_class)


    test_data = gluon.data.DataLoader(
        testset, config.test_batch_size, shuffle=False, last_batch='keep',
        batchify_fn=ms_batchify_fn, num_workers=config.EVAL.workers)

    # create network
    model = get_segmentation_model(num_class=config.num_class,
                                        root=config.model_root,
                                       backbone=config.backbone,norm_layer=mx.gluon.nn.BatchNorm, ctx=config.ctx,
                                       shorter_min=config.EVAL.shorter_min,
                                       shorter_max=config.EVAL.shorter_max)

    # print(model)
    if not config.EVAL.multi_eval:
        evaluator = MultiEvalModel(model, config.num_class, ctx=config.ctx, flip=False, scales=[1.0])
    else:
        evaluator = MultiEvalModel(model, config.num_class, ctx=config.ctx, flip=True, scales=[0.7, 1.0, 1.3])

    metric = SegMetric(config.num_class)


    # load pretrained weight
    assert config.resume is not None, '=> Please provide the checkpoint using --resume'
    if os.path.isfile(config.resume):
        model.load_parameters(config.resume, ctx=config.ctx)
    else:
        raise RuntimeError("=> no checkpoint found at '{}'" \
            .format(config.resume))

    tbar = tqdm(test_data)
    for i, (data, dsts) in enumerate(tbar):
        predicts = [evaluator(data[0].as_in_context(config.ctx))]
        predicts = [pred.argmax(1).asnumpy().squeeze() for pred in predicts]
        targets = [target.as_in_context(mx.cpu()).asnumpy().squeeze() \
                    for target in dsts]
        metric.update(targets, predicts)
       
    pixAcc, mIoU, IoUs = metric.get()
    iou_str = ""
    for ind, cur_class in enumerate(testset.classes):
        iou_str += "%s: %.3f\t" % (cur_class, IoUs[ind])
    logging.info( 'pixAcc: %.4f, mIoU: %.4f\n%s' % (pixAcc, mIoU, iou_str))

if __name__ == "__main__":
    args = parse_args()
    print("Using config file %s" % args.cfg)
    config.update_config(args.cfg)
    config.config.resume = args.resume
    config.config.gpu = args.gpu
    config.config.ctx = args.ctx
    config.config.test_batch_size = 1

    tag = args.cfg.split("/")[-1]
    tag = tag.replace(".yaml", "")
    config.config.tag = tag

    logging.basicConfig(filename=config.config.tag+"_eval.log", level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)

    logging.info("Parameters:")
    logging.info(config.config)

    logging.info('Testing model: %s'%config.config.resume)
    test(config.config)
