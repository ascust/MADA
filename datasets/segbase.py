"""Base segmentation dataset"""
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import mxnet as mx
from mxnet import cpu
import mxnet.ndarray as F
from gluoncv.data.segbase import VisionDataset

__all__ = ['ms_batchify_fn', 'SegmentationDataset']

class SegmentationDataset(VisionDataset):
    """Segmentation Base Dataset"""
    # pylint: disable=abstract-method
    def __init__(self, root, split, mode, transform, ignored_label=255, shorter_min=512, shorter_max=1024,
                 min_scale=0.7, max_scale=1.3, crop_size_w=512, crop_size_h=512, mask_w=None, mask_h=None, random_flip=True, random_gaussian=True):
        super(SegmentationDataset, self).__init__(root)
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.shorter_min = shorter_min
        self.shorter_max = shorter_max
        assert shorter_max >= shorter_min
        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h
        self.ignored_label = ignored_label
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.random_flip = random_flip
        self.random_gaussian = random_gaussian
        self.mask_w = mask_w
        self.mask_h = mask_h

    def set_mask_wh(self, w, h):
        self.mask_w = w
        self.mask_h = h
    def _test_sync_tranform(self, img, mask):
        mask = np.array(mask).astype('int32')
        mask[mask == 255] = self.ignored_label
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _val_sync_transform(self, img, mask):
        return None, None

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5 and self.random_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random scale
        w, h = img.size
        scale = self._get_new_scale(w, h)
        scale *= random.random() * (self.max_scale - self.min_scale) + self.min_scale

        x_start, y_start, new_crop_w, new_crop_h = self._get_crop_params([w, h], scale)
        if w < new_crop_w or h < new_crop_h:
            padh = new_crop_h - h if h < new_crop_h else 0
            padw = new_crop_w - w if w < new_crop_w else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

        img = img.crop((x_start, y_start, x_start+new_crop_w, y_start+new_crop_h))
        mask = mask.crop((x_start, y_start, x_start+new_crop_w, y_start+new_crop_h))

        img = img.resize((self.crop_size_w, self.crop_size_h), Image.BILINEAR)
        if self.mask_h is not None or self.mask_w is not None:
            mask = mask.resize((self.mask_w, self.mask_h), Image.NEAREST)
        else:
            mask = mask.resize((self.crop_size_w, self.crop_size_h), Image.NEAREST)

        # gaussian blur as in PSP
        if random.random() < 0.5 and self.random_gaussian:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        mask = np.array(mask).astype('int32')
        mask[mask == 255] = self.ignored_label
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return F.array(np.array(img), cpu(0))

    def _mask_transform(self, mask):
        return F.array(mask, cpu(0))

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0

    def _get_new_scale(self, w, h):
        if min(w, h) < self.shorter_min:
            return self.shorter_min / float(min(w, h))
        elif min(w, h) > self.shorter_max:
            return self.shorter_max / float(min(w, h)) 
        else:
            return 1.0
    
    def _get_crop_params(self, im_wh, scale):
        im_w, im_h = im_wh

        new_crop_w = int(self.crop_size_w / scale)
        new_crop_h = int(self.crop_size_h / scale)

        x_start = random.randint(0, max(0, im_w-new_crop_w))
        y_start = random.randint(0, max(0, im_h-new_crop_h))
        return x_start, y_start, new_crop_w, new_crop_h

def ms_batchify_fn(data):
    """Multi-size batchfy function"""
    if isinstance(data[0], (str, mx.nd.NDArray)):
        return list(data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [ms_batchify_fn(i) for i in data]
    raise RuntimeError('unknown datatype')
