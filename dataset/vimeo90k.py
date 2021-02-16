'''
vimeo90k dataset
support reading images from lmdb
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data

class Vimeo90kDataset(data.Dataset):
    '''
    Reading the training vimeo90k dataset
    key example: 000_00000000
    GT: [2] Ground-Truth (C, H, W);
    LR: [1, 3] Low resolution (2, C, H, W)
    LR_GT: [2] low resolution frame
    HR_GT: [1, 3] high resolution
    '''
    def __init__(self, data_root, spatial_size=(64, 112), train=True):
        """
        data_root: the folder of vimeo90k lmdb
        spatial size: spatial size for LR image
        train: whether training data or not
        mean: if mean is given, the data will be normalized
        """
        super(Vimeo90kDataset, self).__init__()
        if train:
            self.data_root_GT = osp.join(data_root, 'vimeo90k_train_GT.lmdb')
            self.data_root_LRs = osp.join(data_root, 'vimeo90k_train_LR.lmdb')
        else:
            self.data_root_GT = osp.join(data_root, 'vimeo90k_test_GT.lmdb')
            self.data_root_LRs = osp.join(data_root, 'vimeo90k_test_LR.lmdb')

        self.H, self.W = spatial_size
        self.H_GT, self.W_GT = 4*self.H, 4*self.W
        self.train = train

        
        # directly load image keys
        meta_info = pickle.load(open(osp.join(self.data_root_GT, 'meta_info.pkl'), 'rb'))
        self.paths = sorted(list(meta_info['keys']))
        # self.paths = sorted(list(meta_info['keys']))[:10000]
        assert self.paths, 'Error: path is empty.'

        self.env_GT = lmdb.open(self.data_root_GT, readonly=True, lock=False, readahead=False, meminit=False, sync=False, max_spare_txns=1)
        self.env_LRs = lmdb.open(self.data_root_LRs, readonly=True, lock=False, readahead=False, meminit=False, sync=False, max_spare_txns=1)

    def _read_img_lmdb(self, env, key, size):
        """
        read image from lmdb with key (w/ and w/o fixed size)
        size: (C, H, W) tuple
        return "uint8, (H, W, C), RGB" image
        """
        with env.begin(write=False, buffers=True) as txn:
            buf = txn.get(key.encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        C, H, W = size
        img = img_flat.reshape(H, W, C)
        img = img[:, :, [2, 1, 0]]
        return img

    def _read_pair(self, env, key, size):
        img_1 = self._read_img_lmdb(env, '{}_{}'.format(key, 1), size)
        img_2 = self._read_img_lmdb(env, '{}_{}'.format(key, 2), size)
        img_3 = self._read_img_lmdb(env, '{}_{}'.format(key, 3), size)
        return [img_1, img_2, img_3]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        """
        Read img_pair from img_data
        return
            'LR': 3CHW
            'HR': 3CHW
        """
        key = self.paths[index]
        img_pair_HR = self._read_pair(self.env_GT, key, (3, 256, 448))
        img_pair_LR = self._read_pair(self.env_LRs, key, (3, 64, 112))
        if self.train and random.randint(0, 1):
            img_HR_l = img_pair_HR
            img_LR_l = img_pair_LR
        else:
            img_HR_l = reversed(img_pair_HR)
            img_LR_l = reversed(img_pair_LR)

        # Crop and flip
        if self.train:
            rnd_H = random.randint(0, 64 - self.H)
            rnd_W = random.randint(0, 112 - self.W)
            img_HR_l = [v[4*rnd_H:4*rnd_H+self.H_GT, 4*rnd_W:4*rnd_W+self.W_GT, :] for v in img_HR_l]
            img_LR_l = [v[rnd_H:rnd_H+self.H, rnd_W:rnd_W+self.W, :] for v in img_LR_l]

        if self.train and random.randint(0, 1):
            img_HR_l = [np.fliplr(v) for v in img_HR_l]
            img_LR_l = [np.fliplr(v) for v in img_LR_l]

        if self.train and random.randint(0, 1):
            img_HR_l = [np.flipud(v) for v in img_HR_l]
            img_LR_l = [np.flipud(v) for v in img_LR_l]

        img_HR = np.stack(img_HR_l, axis=0)
        img_LR = np.stack(img_LR_l, axis=0)

        img_HR = torch.from_numpy(np.transpose(img_HR, (0, 3, 1, 2))).float().div(255.)
        img_LR = torch.from_numpy(np.transpose(img_LR, (0, 3, 1, 2))).float().div(255.)

        return {'LR': img_LR, 'HR': img_HR}

