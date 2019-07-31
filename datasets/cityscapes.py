import os
import numpy as np
from PIL import Image

import mxnet as mx
from .segbase import SegmentationDataset

class CitySegmentation(SegmentationDataset):
    """Cityscapes Dataloader"""
    # pylint: disable=abstract-method
    BASE_DIR = 'cityscapes'
    # NUM_CLASS = 19
    def __init__(self, root="../data_root", split='train',
                 mode=None, transform=None, num_sample=None, num_class=19, **kwargs):
        super(CitySegmentation, self).__init__(
            root, split, mode, transform, **kwargs)
        self.NUM_CLASS = num_class
        self.root = os.path.join(root, self.BASE_DIR)

        tmp_imgs, tmp_masks = _get_city_pairs(self.root, self.split)
        tmp_list = zip(tmp_imgs, tmp_masks)
        self.items = []
        if num_sample is None:
            self.items += tmp_list
        else:
            if num_sample <= len(tmp_list):
                self.items += tmp_list
            else:
                for i in range(num_sample/len(tmp_list)+1):
                    np.random.shuffle(tmp_list)
                    self.items += tmp_list

        if len(self.items) == 0:
            raise RuntimeError("Found 0 images!")
        assert num_class in [13, 16, 19]
        if num_class == 19:
            self.mapping = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        elif num_class == 13:
            self.mapping = {7: 0, 8: 1, 11: 2, 19: 3, 20: 4, 21: 5,
                            23: 6, 24: 7, 25: 8, 26: 9, 28: 10, 32: 11, 33: 12}
        elif num_class == 16:
            self.mapping = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6,
                            20: 7, 21: 8, 23: 9, 24: 10, 25: 11, 26: 12,
                            28: 13, 32: 14, 33: 15}

    def __getitem__(self, index):
        img = Image.open(self.items[index][0]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.items[index][0])
        #mask = self.masks[index]
        mask = Image.open(self.items[index][1])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._test_sync_tranform(img, mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def _mask_transform(self, mask):
        target = self.ignored_label * np.ones(mask.shape, dtype=np.float32)
        for k, v in self.mapping.items():
            target[mask == k] = v
        return mx.nd.array(target, mx.cpu(0))

    def __len__(self):
        return len(self.items)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        if self.NUM_CLASS == 19:
            return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcyle',
                    'bicycle')
        elif self.NUM_CLASS == 16:
            return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                    'traffic light', 'traffic sign', 'vegetation', 'sky',
                    'person', 'rider', 'car', 'bus', 'motorcyle',
                    'bicycle')
        
        elif self.NUM_CLASS == 13:
            return ('road', 'sidewalk', 'building',     
                    'traffic light', 'traffic sign', 'vegetation', 'sky',
                    'person', 'rider', 'car', 'bus', 'motorcyle',
                    'bicycle')

def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/'+ split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths

