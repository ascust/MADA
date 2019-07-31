import math
import numpy as np
import mxnet as mx
from mxnet.ndarray import NDArray
from mxnet.gluon.nn import HybridBlock
from gluoncv.utils.parallel import parallel_apply
from .resnet101 import resnet101
from gluoncv.utils.parallel import tuple_map
from .vgg16 import vgg16

def get_segmentation_model(**kwargs):
    from .SegmentationNet import get_net
    return get_net(**kwargs)

class SegBaseModel(HybridBlock):
    # pylint : disable=arguments-differ
    def __init__(self, nclass, backbone='resnet101', height=None, width=None,
                 shorter_min=512, shorter_max=1024, crop_size_w=512, crop_size_h=512, pretrained_base=True, dilated=False, eval_mode=False,**kwargs):
        super(SegBaseModel, self).__init__()
        self.nclass = nclass
        with self.name_scope():
            if backbone == 'resnet101':
                pretrained = resnet101(pretrained=pretrained_base, dilated=dilated, **kwargs)
                self.isvgg = False
            elif backbone == 'vgg16':
                pretrained = vgg16(pretrained=pretrained_base, **kwargs)
                self.isvgg = True
            else:
                raise RuntimeError('unknown backbone: {}'.format(backbone))

            if self.isvgg:
                self.features = pretrained.features
            else:
                self.conv1 = pretrained.conv1
                self.bn1 = pretrained.bn1
                self.relu = pretrained.relu
                self.maxpool = pretrained.maxpool
                self.layer1 = pretrained.layer1
                self.layer2 = pretrained.layer2
                self.layer3 = pretrained.layer3
                self.layer4 = pretrained.layer4
        height = height if height is not None else crop_size_h
        width = width if width is not None else crop_size_w
        self._up_kwargs = {'height': height, 'width': width}
        self.shorter_min = shorter_min
        self.shorter_max = shorter_max
        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h
        self.eval_mode = eval_mode

    def base_forward(self, x):
        """forwarding pre-trained network"""
        if self.isvgg:
            c1 = self.features[:10](x)
            c2 = self.features[10:17](c1)
            c3 = self.features[17:24](c2)
            c4 = self.features[24:](c3)
            return c1, c2, c3, c4

        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            c1 = self.layer1(x)
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)
            c4 = self.layer4(c3)
            return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        out = self.forward(x)[0]
        return out


    def set_wh(self, w, h):
        self._up_kwargs = {'height': h, 'width': w}
        return 0


class MultiEvalModel(object):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, ctx, flip=True,
                 scales=[0.7, 1.0, 1.3]):
        self.flip = flip
        self.shorter_min = module.shorter_min
        self.shorter_max = module.shorter_max
        self.nclass = nclass
        self.scales = scales
        module.collect_params().reset_ctx(ctx=ctx)
        self.evalmodule = module

    def __call__(self, image):
        # only single image is supported for evaluation
        image = image.expand_dims(0)
        batch, _, h, w = image.shape
        assert(batch == 1)
        scores = mx.nd.zeros((batch, self.nclass, h, w), ctx=image.context)
        sc_mult = self._get_new_scale(w, h) 
        for scale in self.scales:
            height = int(scale*sc_mult*h+0.5)
            width = int(scale*sc_mult*w+0.5)
            cur_img = _resize_image(image, height, width)
            outputs = self.flip_inference(cur_img, w, h)
            scores += outputs
        
        scores /= len(self.scales)

        return scores

    def _get_new_scale(self, w, h):
        if min(w, h) < self.shorter_min:
            return self.shorter_min / float(min(w, h))
        elif min(w, h) > self.shorter_max:
            return self.shorter_max / float(min(w, h)) 
        else:
            return 1.0

    def flip_inference(self, image, origw, origh):
        assert(isinstance(image, NDArray))
        self.evalmodule.set_wh(h=origh, w=origw)
        
        output = self.evalmodule(image)[0]
        output = NDArray.softmax(output, axis=1)
        if self.flip:
            fimg = _flip_image(image)
            foutput = self.evalmodule(fimg)[0]
            foutput = NDArray.softmax(foutput, axis=1)
            output = (output + _flip_image(foutput))/2
        return output

    def collect_params(self):
        return self.evalmodule.collect_params()


def _resize_image(img, h, w):
    return mx.nd.contrib.BilinearResize2D(img, height=h, width=w)

def _flip_image(img):
    assert(img.ndim == 4)
    return img.flip(3)
