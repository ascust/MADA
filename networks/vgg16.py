from mxnet.initializer import Xavier
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
import mxnet as mx
class VGG(HybridBlock):
    def __init__(self, layers, filters, **kwargs):
        super(VGG, self).__init__()
        assert len(layers) == len(filters)
        with self.name_scope():
            self.features = self._make_features(layers, filters)
            self.features.add(nn.Conv2D(4096, kernel_size=7, padding=3))
            self.features.add(nn.Activation('relu'))

    def _make_features(self, layers, filters):
        featurizer = nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros'))

                featurizer.add(nn.Activation('relu'))
            featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x

def vgg16(pretrained=False, root='~/.mxnet/models', ctx=mx.cpu(0), **kwargs):
    layers = [2, 2, 3, 3, 3]
    filters = [64, 128, 256, 512, 512]
    model = VGG(layers, filters, **kwargs)
    if pretrained:
        import os
        model.load_parameters(os.path.join(root, "vgg16_nobn.params"), ctx=ctx, allow_missing=True)
        
    return model