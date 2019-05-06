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

num_imgs = 10
ang_interval = 1
ang_skip = 1

rs = np.random.RandomState(123)


class ImgDataset(Dataset):

    def __init__(self, ids, n, name='default',
                 max_examples=None, is_train=True,
                 dataset_name='drc_chair', clamp_elevation=False,
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
            self.bound = int(360/(ang_skip*ang_interval) + 1 - 2)
        else:
            self.bound = int(360/ang_interval + 1)

        if self.use_file_list:
            with open(osp.join('testing_tuple_lists/id_' + dataset_name + '_random_elevation.txt'), 'r') as fp:
                self.ids_files     = [s.strip() for s in fp.readlines() if s]
                self.ids_files_tgt = [s.split(' ')[0] for s in self.ids_files if s]
                self.ids_files_src = [s.split(' ')[1] for s in self.ids_files if s]
                self.ids_files_all = [s.split(' ')    for s in self.ids_files if s]
        else:
            self.ids_files     = None
            self.ids_files_tgt = None
            self.ids_files_src = None
            self.ids_files_all = None

        self.concat_mask = concat_mask
        self.random_pairs = random_pairs
        self.clamp_elevation = clamp_elevation
        self.img_path = img_path

        self.rotate_increment = 360
        self.num_elevations = 3
        self.elev_increment = 1.0

        self.input_width  = input_width
        self.input_height = input_height

        self.transform = nn.Upsample(size=[self.input_height, self.input_width], mode='bilinear')

        if self.random_pairs:
            self.num_pair_samples = 1
        else:
            self.num_pair_samples = 2

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

    def __getitem__(self, in_id):
        if self.random_pairs:
            imgs_to_gen = num_imgs
            if self.args.cull_identity_transform:
                lower_bound = 1
            else:
                lower_bound = 0
            idx = np.arange(lower_bound, imgs_to_gen)
            np.random.shuffle(idx)
            idx = idx[:self.n]

            prefix_in_id = int(in_id / num_imgs)
            suffix_in_id = int(in_id % num_imgs)
        else:
            imgs_to_gen = num_imgs
            idx = np.arange(0, imgs_to_gen)
            idx = idx[:self.n]

            prefix_in_id = int(in_id / self.num_pair_samples)
            suffix_in_id = int(in_id % self.num_pair_samples)

        suffix_in_id_str = '/render_' + str(suffix_in_id)
        id = self._ids[prefix_in_id]

        int_tgt = suffix_in_id

        ang = (idx + int_tgt).astype(np.int32)

        return self.get_data_pair(id, ang, int_tgt, suffix_in_id_str)

    def get_data_pair(self, id, ang, int_tgt, suffix_in_id_str):
        image, mask = self.readImageToArray(id + suffix_in_id_str)
        image[0.0 == mask] = [1.0, 1.0, 1.0]
        if self.concat_mask:
            image = np.concatenate((image, np.expand_dims(mask, -1)), axis=-1)

        with open(osp.join(self.img_path, self.dataset_name + '/' + id + '/' + 'view.txt' ), 'r') as fp:
            img_pos_params = [s.strip() for s in fp.readlines() if s]

        tgt_params = img_pos_params[int_tgt].split(' ')
        azimuth   = float(tgt_params[0])
        elevation = float(tgt_params[1])

        tgt_azim_transform_mode      =   azimuth
        tgt_elev_transform_mode = elevation

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

        for a in ang:
            int_src = a % num_imgs

            id_src = (id + suffix_in_id_str)[:-1] + str(int_src)
            image_tmp, mask_tmp = self.readImageToArray(id_src)
            image_tmp[0.0 == mask_tmp] = np.array([1.0, 1.0, 1.0])

            if self.concat_mask:
                image_tmp = np.concatenate((image_tmp, np.expand_dims(mask_tmp, -1)), axis=-1)

            src_params = img_pos_params[int_src].split(' ')
            azimuth = float(src_params[0])
            elevation = float(src_params[1])

            src_azim_transform_mode = azimuth
            src_elev_transform_mode = elevation

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
        if not self.random_pairs:
            return self.num_pair_samples * len(self.ids)

        return num_imgs * self.num_pair_samples * len(self.ids)

    def __repr__(self):
        return 'ImgDataset (%s, %d examples)' % (
            self.name,
            len(self)
        )

    def readImageToArray(self, in_id):
        img = np.array(Image.open(self.img_path + '/' + self.dataset_name + '/' + in_id + '.png'))/255.0

        rgb   = img[:, :, 0:3]
        alpha = img[:, :, 3  ]
        alpha = np.expand_dims(alpha, 2)

        # apply alpha channel
        rgb = alpha * rgb + (1.0 - alpha) * np.array([1.0, 1.0, 1.0])

        # make alpha channel into binary mask
        mask  = np.zeros(alpha.shape)
        mask[0.0 == alpha] = 0.0
        mask[0.0 != alpha] = 1.0
        mask = mask.squeeze()

        return rgb, mask

def create_default_splits(n, is_train=True, dataset_name='drc_chair',
                          input_width=80, input_height=80, concat_mask=True,
                          shuffle_train=True, shuffle_test=True, img_path='./datasets/shapenet', args=None):
    ids_train, ids_test, sorted_ids_val = all_ids(dataset_name=dataset_name, shuffle_train=shuffle_train, shuffle_test=shuffle_test)

    print('concat_mask: ' + str(concat_mask))
    dataset_train = ImgDataset(ids_train, n, name='train', is_train=is_train, dataset_name=dataset_name,
                               input_width=input_width, input_height=input_height, concat_mask=concat_mask,
                               img_path=img_path, args=args)
    dataset_test  = ImgDataset(ids_test , n, name='test' , is_train=is_train, dataset_name=dataset_name,
                               input_width=input_width, input_height=input_height, concat_mask=concat_mask,
                               img_path=img_path, args=args)
    dataset_file_val = ImgDataset(sorted_ids_val, n, name='file_test', is_train=is_train, dataset_name=dataset_name,
                                   input_width=input_width, input_height=input_height, concat_mask=concat_mask,
                                   img_path=img_path, use_file_list=False, random_pairs=False, args=args)
    return dataset_train, dataset_test, dataset_file_val


def all_ids(dataset_name='drc_chair', shuffle_train=True, shuffle_test=True):

    with open(osp.join(__PATH__, dataset_name + '_train.txt'), 'r') as fp:
        ids_train = [s.strip() for s in fp.readlines() if s]
    if shuffle_train:
        rs.shuffle(ids_train)

    with open(osp.join(__PATH__, dataset_name + '_test.txt' ), 'r') as fp:
        ids_test = [s.strip() for s in fp.readlines() if s]
    if shuffle_test:
        rs.shuffle(ids_test)

    with open(osp.join(__PATH__, dataset_name + '_val.txt'  ), 'r') as fp:
        sorted_ids_val = [s.strip() for s in fp.readlines() if s]

    return ids_train, ids_test, sorted_ids_val
