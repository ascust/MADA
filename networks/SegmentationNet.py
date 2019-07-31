from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from mxnet import gluon
from .segbase import SegBaseModel

class SegmentationNet(SegBaseModel):

    def __init__(self, nclass, root, norm_layer, backbone='resnet101', ctx=cpu(), pretrained_base=True,
                 base_size=520, crop_size=480, new_layer_multi=False, norm_kwargs={}, **kwargs):
        super(SegmentationNet, self).__init__(nclass, root=root, backbone=backbone, ctx=ctx, base_size=base_size,
                                     crop_size=crop_size, pretrained_base=pretrained_base, new_layer_multi=new_layer_multi, 
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs, dilated=False, **kwargs)
        with self.name_scope():
            if backbone == "vgg16":
                self.merge1 = _MergeBlock(4096, 512, 128, 3, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
                self.merge2 = _MergeBlock(128, 256, 128, 6, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
                self.merge3 = _MergeBlock(128, 128, 128, 12, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
            else:
                self.merge1 = _MergeBlock(2048, 1024, 128, 3, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
                self.merge2 = _MergeBlock(128, 512, 128, 6, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
                self.merge3 = _MergeBlock(128, 256, 128, 12, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

            self.final = nn.HybridSequential()
            with self.final.name_scope():
                self.final.add(norm_layer(in_channels=128, **norm_kwargs))
                self.final.add(nn.Activation('relu'))

            self.classifier = _AsppBlock(nclass)
            self.merge1.initialize(ctx=ctx)
            self.merge2.initialize(ctx=ctx)
            self.merge3.initialize(ctx=ctx)
            self.final.initialize(ctx=ctx)
            self.classifier.initialize(ctx=ctx)
            if new_layer_multi:
                self.merge1.collect_params().setattr('lr_mult', 10)
                self.merge2.collect_params().setattr('lr_mult', 10)
                self.merge3.collect_params().setattr('lr_mult', 10)
                self.final.collect_params().setattr('lr_mult', 10)
                self.classifier.collect_params().setattr('lr_mult', 10)

    def hybrid_forward(self, F, x):
        c1, c2, c3, c4 = self.base_forward(x)
        merge1 = self.merge1(c4, c3)
        merge2 = self.merge2(merge1, c2)
        merge3 = self.merge3(merge2, c1)
        feat = self.final(merge3)
        pred = self.classifier(feat)
        pred = F.contrib.BilinearResize2D(pred, **self._up_kwargs)

        return (pred, feat, c3)

class _MergeBlock(HybridBlock):
    def __init__(self, in1_channels, in2_channels, out_channels, atrous_rate, norm_layer, norm_kwargs={}, **kwargs):
        super(_MergeBlock, self).__init__()
        with self.name_scope():
            self.adapt1 = _Unit(in_channels=in1_channels, out_channels=out_channels,
                                atrous_rate=atrous_rate, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
            self.adapt2 = nn.Conv2D(in_channels=in2_channels, channels=out_channels,
                            kernel_size=1, padding=0, use_bias=True)
    
    def upsample(self, F, x, h, w):
        return F.contrib.BilinearResize2D(x, height=h, width=w)

    def hybrid_forward(self, F, x1, x2):
        _, _, h, w = x2.shape
        res1 = self.upsample(F, self.adapt1(x1), h, w)
        res2 = self.adapt2(x2)
        return res1 + res2
class _AsppBlock(HybridBlock):
    def __init__(self, nclass):
        super(_AsppBlock, self).__init__()
        with self.name_scope():
            self.b1 = nn.Conv2D(in_channels=128, channels=nclass,
                                     kernel_size=3, strides=1, dilation=1, padding=1)
            self.b2 = nn.Conv2D(in_channels=128, channels=nclass,
                                     kernel_size=3, strides=1, dilation=2, padding=2)
            self.b3 = nn.Conv2D(in_channels=128, channels=nclass,
                                     kernel_size=3, strides=1, dilation=4, padding=4)
            self.b4 = nn.Conv2D(in_channels=128, channels=nclass,
                                     kernel_size=3, strides=1, dilation=8, padding=8)
    def hybrid_forward(self, F, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        return b1 + b2 + b3 + b4


def _Unit(in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs={}, **kwargs):
    block = nn.HybridSequential()
    with block.name_scope():
        block.add(norm_layer(in_channels=in_channels, **norm_kwargs))
        block.add(nn.Activation('relu'))
        block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                            kernel_size=3, padding=atrous_rate,
                            dilation=atrous_rate, use_bias=True))
    return block


def get_net(num_class, backbone='resnet101', pretrained=False, ctx=cpu(0), **kwargs):
    model = SegmentationNet(num_class, backbone=backbone, ctx=ctx, **kwargs)
    return model
