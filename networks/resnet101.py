from __future__ import division

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from gluoncv.model_zoo.model_store import get_model_file
from gluoncv.data import ImageNet1kAttr
import os

class Bottleneck(HybridBlock):

    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, planes,
                 downsample=None, norm_layer=None, strides=1,
                 norm_kwargs={}, **kwargs):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(channels=planes, kernel_size=1,
                               use_bias=False)
        self.bn1 = norm_layer(**norm_kwargs)
        self.relu1 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(channels=planes, kernel_size=3, strides=strides,
                               padding=1, use_bias=False)
        self.bn2 = norm_layer(**norm_kwargs)
        self.relu2 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(channels=planes * 4, kernel_size=1, use_bias=False)
        self.bn3 = norm_layer(**norm_kwargs)

        self.relu3 = nn.Activation('relu')
        self.downsample = downsample
        self.strides = strides

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu3(out)

        return out

class ResNet(HybridBlock):

    # pylint: disable=unused-variable
    def __init__(self, block, layers, classes=1000, norm_layer=BatchNorm,
                 norm_kwargs={}, use_global_stats=False,
                 name_prefix='', **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__(prefix=name_prefix)
        self.norm_kwargs = norm_kwargs
        if use_global_stats:
            self.norm_kwargs['use_global_stats'] = True
        with self.name_scope():

            self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                    padding=3, use_bias=False)

            self.bn1 = norm_layer(**norm_kwargs)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.layer1 = self._make_layer(1, block, 64, layers[0], norm_layer=norm_layer)
            self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2, norm_layer=norm_layer)
            self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2, norm_layer=norm_layer)

            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.fc = nn.Dense(in_units=512 * block.expansion, units=classes)

    def _make_layer(self, stage_index, block, planes, blocks, strides=1,
                    norm_layer=None):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='down%d_'%stage_index)
            with downsample.name_scope():
                downsample.add(nn.Conv2D(channels=planes * block.expansion,
                                            kernel_size=1, strides=strides, use_bias=False))
                downsample.add(norm_layer(**self.norm_kwargs))

        layers = nn.HybridSequential(prefix='layers%d_'%stage_index)
        with layers.name_scope():
            layers.add(block(planes, strides=strides,
                                downsample=downsample,
                                norm_layer=norm_layer, norm_kwargs=self.norm_kwargs))


        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.add(block(planes, norm_layer=norm_layer,
                                norm_kwargs=self.norm_kwargs))

        return layers

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc(x)

        return x



def resnet101(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):

    model = ResNet(Bottleneck, [3, 4, 23, 3], name_prefix='resnet_', **kwargs)
    if pretrained:

        model.load_parameters(os.path.join(root, "resnet101.params"), ctx=ctx)

        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model