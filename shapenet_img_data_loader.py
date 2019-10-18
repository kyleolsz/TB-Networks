from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

__PATH__ = './datasets/shapenet'
ang_interval = 10
ang_skip = 2
rs = np.random.RandomState(123)


class ImgDataset(Dataset):

    def __init__(self, ids, n, name='default',
                 max_examples=None, is_train=True,
                 dataset_name='chair', clamp_elevation=False,
                 input_width=80, input_height=80, concat_mask=False,
                 img_path='./datasets/shapenet', random_pairs=True,
                 use_file_list=False, args=None):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.n = n
        self.args = args
        self.use_file_list = use_file_list

        if self.args.cull_identity_transform:
            self.bound = int(360 / (ang_skip * ang_interval) + 1 - 2)
        else:
            self.bound = int(360 / ang_interval + 1)

        if self.use_file_list:
            with open(osp.join('testing_tuple_lists/id_' + dataset_name + '_random_elevation.txt'), 'r') as fp:
                self.ids_files = [s.strip() for s in fp.readlines() if s]
                self.ids_files_tgt = [s.split(' ')[0] for s in self.ids_files if s]
                self.ids_files_src = [s.split(' ')[1] for s in self.ids_files if s]
                self.ids_files_all = [s.split(' ') for s in self.ids_files if s]
        else:
            self.ids_files = None
            self.ids_files_tgt = None
            self.ids_files_src = None
            self.ids_files_all = None

        self.concat_mask = concat_mask
        self.use_tgt_bg_noise = False
        self.use_src_bg_noise = False
        self.random_pairs = random_pairs
        self.clamp_elevation = clamp_elevation
        self.img_path = img_path

        self.rotate_increment = 36
        self.num_elevations = 3
        self.elev_increment = 10.0

        self.input_width = input_width
        self.input_height = input_height

        self.transform = nn.Upsample(size=[self.input_height, self.input_width], mode='bilinear')

        if self.random_pairs:
            self.num_pair_samples = 1
        else:
            self.num_pair_samples = int(self.rotate_increment / ang_skip)

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

    def __getitem__(self, in_id):
        if self.use_file_list:
            id = self.ids_files_all[in_id][0]

            int_tgt = int(id.split('_')[1])
            h = id.split('_')[-1]
            int_elev_src = [int(h)]*self.n
            int_elev_tgt = [int(h)]

            ang = []
            for src_idx in range(self.n):
                ang.append(int(self.ids_files_all[in_id][src_idx + 1].split('_')[1]))

            return self.get_data_pair(id, ang, int_tgt, int_elev_src, int_elev_tgt)

        if self.clamp_elevation:
            sample_interval = self.num_elevations
        else:
            sample_interval = 1

        if not self.random_pairs:
            id = self._ids[sample_interval * int(in_id / self.num_pair_samples)]

            idx = np.array([in_id % self.num_pair_samples]) * ang_skip

            id_base = id.split('_')[0]
            tgt = id.split('_')[1]
            h = id.split('_')[-1]

            if self.clamp_elevation:
                new_h = '0'
            else:
                new_h = h

            id = '_'.join([id_base, tgt, new_h])

            int_tgt = int(tgt)
            int_elev_tgt = int(new_h)

            ang = (idx + int_tgt).astype(np.int32)

            return self.get_data_pair(id, ang, int_tgt, int_elev_tgt, new_h)

        id = self._ids[sample_interval * in_id]

        elev_transform = False

        if self.args.use_elev_transform and np.random.uniform(0, 1, 1)[0] < self.args.elev_transform_threshold:
            elev_transform = True
        else:
            elev_transform = False

        id_base = id.split('_')[0]
        tgt = id.split('_')[1]
        h = id.split('_')[-1]

        if self.clamp_elevation:
            new_h = '0'
        else:
            new_h = h

        id = '_'.join([id_base, tgt, new_h])
        int_tgt = int(tgt)

        idx = np.concatenate(
            (np.linspace(-self.bound, 0, self.bound + 1)[:-1],
             np.linspace(0, self.bound, self.bound + 1)[1:])
        ) * ang_skip

        np.random.shuffle(idx)
        idx = idx[:self.n]
        h = str(new_h)

        if elev_transform:
            int_elev_src = 10 * np.random.randint(0, 3, self.n)
            int_elev_tgt = [int(new_h)]
        else:
            int_elev_src = [int(new_h)]*self.n
            int_elev_tgt = [int(new_h)]

        ang = (idx + int_tgt).astype(np.int32)

        return self.get_data_pair(id, ang, int_tgt, int_elev_src, int_elev_tgt)

    def get_data_pair(self, id, ang, int_tgt, int_elev_src, int_elev_tgt):
        image = self.readImageToArray(id)
        mask = 1 - (np.sum(image, axis=-1) >= 2.997)
        if self.args.use_tgt_bg_noise:
            noise_image = np.random.normal(0.5, 0.5, (256, 256, 3))
            torch.clamp(noise_image, min=0.0, max=1.0)

            rgb_mask = np.expand_dims(mask, -1)
            rgb_mask = np.concatenate((rgb_mask, rgb_mask, rgb_mask), axis=-1)
            image = ((rgb_mask) * (image)) + ((1 - rgb_mask) * (noise_image))
        if self.concat_mask:
            image = np.concatenate((image, np.expand_dims(mask, -1)), axis=-1)

        tgt_azim_transform_mode = (self.rotate_increment - int_tgt) % self.rotate_increment
        tgt_elev_transform_mode = int(int_elev_tgt[0] / self.elev_increment)

        data = {}

        tgt_image = torch.Tensor(image.transpose(2, 0, 1))[:, self.args.crop_y_dim:(self.args.final_height-self.args.crop_y_dim),
                                                              self.args.crop_x_dim:(self.args.final_width-self.args.crop_x_dim)]
        tgt_mask  = torch.Tensor(mask).unsqueeze(0)[:, self.args.crop_y_dim:(self.args.final_height-self.args.crop_y_dim),
                                                       self.args.crop_x_dim:(self.args.final_width-self.args.crop_x_dim)]
        data['tgt_rgb_image'] = [self.transform(tgt_image.unsqueeze(0)).squeeze(0)]
        data['tgt_seg_image'] = [self.transform(tgt_mask.unsqueeze(0)).squeeze(0)]
        if self.args.upsample_output:
            data['orig_tgt_rgb_image'] = [tgt_image]
            data['orig_tgt_seg_image'] = [tgt_mask]

        data['tgt_azim_transform_mode'] = [tgt_azim_transform_mode]
        data['tgt_elev_transform_mode'] = [tgt_elev_transform_mode]

        data['src_rgb_image'] = []
        data['src_seg_image'] = []
        data['src_azim_transform_mode'] = []
        data['src_elev_transform_mode'] = []

        for idx, a in enumerate(ang):
            id_base = id.split('_')[0]

            int_src = a % self.rotate_increment

            src = str(int_src)

            id_src = '_'.join([id_base, src, str(int_elev_src[idx])])
            image_tmp = self.readImageToArray(id_src)
            mask_tmp = 1 - (np.sum(image_tmp, axis=-1) >= 2.997)

            if self.args.use_src_bg_noise:
                noise_image = np.random.normal(0.5, 0.5, (256, 256, 3))
                torch.clamp(noise_image, min=0.0, max=1.0)

                rgb_mask_tmp = np.expand_dims(mask_tmp, -1)
                rgb_mask_tmp = np.concatenate((rgb_mask_tmp, rgb_mask_tmp, rgb_mask_tmp), axis=-1)
                image_tmp = ((rgb_mask_tmp) * (image_tmp)) + ((1 - rgb_mask_tmp) * (noise_image))
            if self.concat_mask:
                image_tmp = np.concatenate((image_tmp, np.expand_dims(mask_tmp, -1)), axis=-1)

            src_azim_transform_mode = (self.rotate_increment - int_src) % self.rotate_increment
            src_elev_transform_mode = int(int_elev_src[idx] / self.elev_increment)

            src_image = torch.Tensor(image_tmp.transpose(2, 0, 1))[:, self.args.crop_y_dim:(self.args.final_height-self.args.crop_y_dim),
                                                                      self.args.crop_x_dim:(self.args.final_width-self.args.crop_x_dim)]
            src_mask  = torch.Tensor(mask_tmp).unsqueeze(0)[:, self.args.crop_y_dim:(self.args.final_height-self.args.crop_y_dim),
                                                               self.args.crop_x_dim:(self.args.final_width-self.args.crop_x_dim)]
            data['src_rgb_image'].append(self.transform(src_image.unsqueeze(0)).squeeze(0))
            data['src_seg_image'].append(self.transform(src_mask.unsqueeze(0)).squeeze(0))

            data['src_azim_transform_mode'].append(src_azim_transform_mode)
            data['src_elev_transform_mode'].append(src_elev_transform_mode)

        return data

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        if self.use_file_list:
            return len(self.ids_files)

        if self.clamp_elevation:
            return self.num_pair_samples * int(len(self.ids) / self.num_elevations)
        return self.num_pair_samples * len(self.ids)

    def __repr__(self):
        return 'ImgDataset (%s, %d examples)' % (
            self.name,
            len(self)
        )

    def readImageToArray(self, in_id):
        img = np.array(Image.open(self.img_path + '/' + self.dataset_name + '/' + in_id + '.png')) / 255.0
        return img


def create_default_splits(n, is_train=True, dataset_name='chair',
                          input_width=80, input_height=80, concat_mask=False,
                          shuffle_train=True, shuffle_test=True, img_path='./datasets/shapenet', args=None):
    ids_train, ids_test = all_ids(dataset_name=dataset_name, shuffle_train=shuffle_train, shuffle_test=shuffle_test)

    dataset_train = ImgDataset(ids_train, n, name='train', is_train=is_train, dataset_name=dataset_name,
                               input_width=input_width, input_height=input_height, concat_mask=concat_mask,
                               img_path=img_path, args=args)
    dataset_test = ImgDataset(ids_test, n, name='test', is_train=is_train, dataset_name=dataset_name,
                              input_width=input_width, input_height=input_height, concat_mask=concat_mask,
                              img_path=img_path, args=args)
    dataset_file_test = ImgDataset(ids_test, n, name='file_test', is_train=is_train, dataset_name=dataset_name,
                                   input_width=input_width, input_height=input_height, concat_mask=concat_mask,
                                   img_path=img_path, use_file_list=True, args=args)
    return dataset_train, dataset_test, dataset_file_test


def all_ids(dataset_name='chair', shuffle_train=True, shuffle_test=True):
    with open(osp.join(__PATH__, 'id_' + dataset_name + '_train.txt'), 'r') as fp:
        ids_train = [s.strip() for s in fp.readlines() if s]
    if shuffle_train:
        rs.shuffle(ids_train)

    with open(osp.join(__PATH__, 'id_' + dataset_name + '_test.txt'), 'r') as fp:
        ids_test = [s.strip() for s in fp.readlines() if s]
    if shuffle_test:
        rs.shuffle(ids_test)

    return ids_train, ids_test
