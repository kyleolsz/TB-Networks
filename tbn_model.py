from __future__ import absolute_import, division, print_function
import numpy as np
import math
from volume_sampler import apply_volume_transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T
import scipy.io as sio


class Conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride, is_3d_conv=False, dilation=1,
                 use_normalization=True,
                 use_relu=False):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        self.is_3d_conv = is_3d_conv
        self.dilation = dilation
        self.use_normalization = use_normalization
        self.use_relu = use_relu

        if not self.is_3d_conv:
            self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                                       dilation=self.dilation)
            if self.use_normalization:
                self.normalize = nn.BatchNorm2d(num_out_layers)
        else:
            self.conv_base = nn.Conv3d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                                       dilation=self.dilation)
            if self.use_normalization:
                self.normalize = nn.BatchNorm3d(num_out_layers)

    def forward(self, x):
        p = int(np.floor(self.dilation * (self.kernel_size - 1) / 2))
        if not self.is_3d_conv:
            pd = (p, p, p, p)
        else:
            pd = (p, p, p, p, p, p)
        x = self.conv_base(F.pad(x, pd))
        if self.use_normalization:
            x = self.normalize(x)
        if self.use_relu:
            return F.relu(x, inplace=True)
        else:
            return F.elu(x, inplace=True)


class ResConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride, kernel_size=3, is_3d_conv=False):
        super(ResConv, self).__init__()
        self.is_3d_conv = is_3d_conv
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = Conv(num_in_layers, num_out_layers, 1, 1, self.is_3d_conv)
        self.conv2 = Conv(num_out_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                          is_3d_conv=self.is_3d_conv)
        if not self.is_3d_conv:
            self.conv3 = nn.Conv2d(num_out_layers, 4 * num_out_layers, kernel_size=1, stride=1)
            self.conv4 = nn.Conv2d(num_in_layers, 4 * num_out_layers, kernel_size=1, stride=stride)
            self.normalize = nn.BatchNorm2d(4 * num_out_layers)
        else:
            self.conv3 = nn.Conv3d(num_out_layers, 4 * num_out_layers, kernel_size=1, stride=1)
            self.conv4 = nn.Conv3d(num_in_layers, 4 * num_out_layers, kernel_size=1, stride=stride)
            self.normalize = nn.BatchNorm3d(4 * num_out_layers)

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        shortcut = self.conv4(x)
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


def ResBlock(num_in_layers, num_out_layers, num_blocks, stride, kernel_size=3, is_3d_conv=False):
    layers = [
        ResConv(num_in_layers, num_out_layers, stride, kernel_size=kernel_size, is_3d_conv=is_3d_conv)
    ]

    for i in range(1, num_blocks - 1):
        layers.append(
            ResConv(4 * num_out_layers, num_out_layers, 1, kernel_size=kernel_size, is_3d_conv=is_3d_conv)
        )

    layers.append(
        ResConv(4 * num_out_layers, num_out_layers, 1, kernel_size=kernel_size, is_3d_conv=is_3d_conv)
    )
    return nn.Sequential(*layers)


class UpConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale, is_3d_conv=False):
        super(UpConv, self).__init__()
        self.is_3d_conv = is_3d_conv
        self.up_nn = nn.Upsample(scale_factor=scale)
        self.conv1 = Conv(num_in_layers, num_out_layers, kernel_size, 1, is_3d_conv=is_3d_conv)

    def forward(self, x):
        x = self.up_nn(x)
        return self.conv1(x)


class OutputConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers=3, is_3d_conv=False, kernel_size=3):
        super(OutputConv, self).__init__()
        self.is_3d_conv = is_3d_conv
        self.kernel_size = kernel_size
        self.sigmoid = torch.nn.Sigmoid()
        if not self.is_3d_conv:
            self.conv1 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=self.kernel_size, stride=1)
        else:
            self.conv1 = nn.Conv3d(num_in_layers, num_out_layers, kernel_size=self.kernel_size, stride=1)

    def forward(self, x):
        if self.kernel_size > 1:
            p = 1
            if not self.is_3d_conv:
                pd = (p, p, p, p)
            else:
                pd = (p, p, p, p, p, p)
            x = self.conv1(F.pad(x, pd))
        else:
            x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class TBN(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, args, vol_dim=40, num_features=800,
                 tensor_type='torch.FloatTensor', ):
        super(TBN, self).__init__()

        self.args = args
        self.vol_dim = vol_dim
        self.num_features = num_features
        self.tensor_type = tensor_type
        self.num_enc_features = int(self.args.num_gen_features/self.args.encode_feature_scale_factor)
        self.num_dec_features = int(self.args.num_gen_features/self.args.decode_feature_scale_factor)

        in_layers = num_in_layers
        if 0 < self.args.num_input_convs:
            init_num_in_layers = num_in_layers
            middle_num_in_layers = self.num_enc_features
            middle_num_out_layers = middle_num_in_layers
            final_num_out_layers = middle_num_in_layers
            in_layers = final_num_out_layers
            in_conv_layers = []

            for idx in range(self.args.num_input_convs):
                if 0 == idx:
                    conv_in_layers = init_num_in_layers
                    conv_out_layers = middle_num_in_layers
                elif self.args.num_input_convs - 1 == idx:
                    conv_in_layers = middle_num_in_layers
                    conv_out_layers = middle_num_out_layers
                else:
                    conv_in_layers = middle_num_in_layers
                    conv_out_layers = final_num_out_layers

                in_conv_layers.append(
                    Conv(num_in_layers=conv_in_layers, num_out_layers=conv_out_layers, kernel_size=4, stride=2,
                         is_3d_conv=False, dilation=1, use_normalization=True)
                )

            self.in_conv = nn.Sequential(*in_conv_layers)

        self.conv1_2d_encode = Conv(in_layers, 2 * self.num_enc_features, 7, 2)
        self.conv2_2d_encode = ResBlock(2 * self.num_enc_features, self.num_enc_features, self.args.num_res_convs, 2)
        self.conv3_2d_encode = ResBlock(4 * self.num_enc_features, 2 * self.num_enc_features, self.args.num_res_convs, 2)
        self.conv4_2d_encode = ResBlock(8 * self.num_enc_features, 4 * self.num_enc_features, self.args.num_res_convs, 2)

        self.upconv4_2d_encode = UpConv(16 * self.num_enc_features, 8 * self.num_enc_features, 3, 2)
        self.iconv4_2d_encode = Conv(2 * 8 * self.num_enc_features, 8 * self.num_enc_features, 3, 1)

        self.upconv3_2d_encode = UpConv(8 * self.num_enc_features, 4 * self.num_enc_features, 3, 2)
        self.iconv3_2d_encode = Conv(2 * 4 * self.num_enc_features, 4 * self.num_enc_features, 3, 1)

        self.upconv2_2d_encode = UpConv(4 * self.num_enc_features, self.num_features, 3, 2)
        self.iconv2_2d_encode = Conv(2 * self.num_enc_features + self.num_features, self.num_features, 3, 1)

        self.src_seg2d = OutputConv(self.num_features, 1) if 0.0 < self.args.w_gen_seg3d else None

        self.conv2_2d_decode = ResBlock(self.num_features, self.num_dec_features, self.args.num_res_convs, 2)
        self.conv3_2d_decode = ResBlock(4 * self.num_dec_features, 2 * self.num_dec_features, self.args.num_res_convs, 2)
        self.conv4_2d_decode = ResBlock(8 * self.num_dec_features, 4 * self.num_dec_features, self.args.num_res_convs, 2)

        self.upconv4_2d_decode = UpConv(16 * self.num_dec_features, 8 * self.num_dec_features, 3, 2)
        self.iconv4_2d_decode = Conv(2 * 8 * self.num_dec_features, 8 * self.num_dec_features, 3, 1)

        self.upconv3_2d_decode = UpConv(8 * self.num_dec_features, 4 * self.num_dec_features, 3, 2)
        self.iconv3_2d_decode = Conv(2 * 4 * self.num_dec_features, 4 * self.num_dec_features, 3, 1)

        self.upconv2_2d_decode = UpConv(4 * self.num_dec_features, 2 * self.num_dec_features, 3, 2)
        self.iconv2_2d_decode = Conv(self.num_features + 2 * self.num_dec_features, 2 * self.num_dec_features, 3, 1)

        self.upconv1_2d_decode = UpConv(2 * self.num_dec_features, self.num_dec_features, 3, 2)
        self.iconv1_2d_decode = Conv(self.num_dec_features, self.num_dec_features, 3, 1)

        if 0 < self.args.num_output_deconvs:
            deconv_layers = []
            init_num_in_layers = self.num_dec_features
            middle_num_in_layers = self.num_dec_features
            middle_num_out_layers = middle_num_in_layers
            final_num_out_layers = middle_num_in_layers

            for idx in range(self.args.num_output_deconvs):
                if 0 == idx:
                    conv_in_layers = init_num_in_layers
                    conv_out_layers = middle_num_in_layers
                elif self.args.num_output_deconvs - 1 == idx:
                    conv_in_layers = middle_num_in_layers
                    conv_out_layers = middle_num_out_layers
                else:
                    conv_in_layers = middle_num_in_layers
                    conv_out_layers = final_num_out_layers
                deconv_layers.append(UpConv(conv_in_layers, conv_out_layers, 3, 2))

            self.deconv = nn.Sequential(*deconv_layers)

        self.output_2d_decode = OutputConv(self.num_dec_features, num_out_layers)

        num_3d_features = int(self.num_features / self.vol_dim)
        self.conv1_3d_encode = Conv(num_3d_features, self.num_enc_features, 3, 1, is_3d_conv=True)
        self.conv2_3d_encode = Conv(self.num_enc_features, num_3d_features, 3, 1, is_3d_conv=True)

        self.conv1_3d_decode = Conv(num_3d_features, self.num_dec_features, 3, 1, is_3d_conv=True)
        self.conv2_3d_decode = Conv(self.num_dec_features, num_3d_features, 3, 1, is_3d_conv=True)

        if self.args.use_seg3d_proxy:
            self.init_seg3d_proxy(num_3d_features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)

    def init_seg3d_proxy(self, num_3d_features):
        self.seg_conv1_3d_encode = Conv(num_3d_features, int(self.args.num_gen_features/self.args.encode_feature_scale_factor), 3, 1, is_3d_conv=True)
        self.seg_conv2_3d_encode = Conv(int(self.args.num_gen_features/self.args.encode_feature_scale_factor), num_3d_features, 3, 1, is_3d_conv=True)
        self.seg_conv1_3d_decode = Conv(num_3d_features, int(self.args.num_gen_features/self.args.decode_feature_scale_factor), 3, 1, is_3d_conv=True)
        self.seg_conv2_3d_decode = Conv(int(self.args.num_gen_features/self.args.decode_feature_scale_factor), 1, 3, 1, is_3d_conv=True)

        self.seg_conv_3d_final = nn.Softmax(dim=2) if self.args.use_seg3d_softmax else nn.Sigmoid()

        self.seg_conv_3d = nn.Sequential(self.seg_conv1_3d_encode, self.seg_conv2_3d_encode,
                                         self.seg_conv1_3d_decode, self.seg_conv2_3d_decode,
                                         self.seg_conv_3d_final)

        self.seg_conv_3d_to_2d = OutputConv(self.vol_dim, 1, is_3d_conv=False, kernel_size=1)
        self.seg_inv_transform = nn.Upsample(size=[self.args.input_height, self.args.input_width], mode='nearest')

        self.seg_conv_3d_combine = nn.Sequential(self.seg_conv_3d_to_2d, self.seg_inv_transform)

        self.seg_transform = nn.Upsample(size=[self.vol_dim, self.vol_dim], mode='bilinear')

    def get_flow_fields(self, crnt_transform_mode, final_elev_transform_mode, init_elev_transform_mode,
                        width, height, depth, encoded_3d_vol):
        rotation_angle = self.args.azim_rotation_angle_increment * crnt_transform_mode
        init_elev_rotation_angle = self.args.elev_rotation_angle_increment * init_elev_transform_mode
        final_elev_rotation_angle = self.args.elev_rotation_angle_increment * final_elev_transform_mode

        rotation_radians = (np.pi / 180.0) * rotation_angle.type(self.tensor_type)
        init_elev_rotation_radians = (np.pi / 180.0) * init_elev_rotation_angle.type(self.tensor_type)
        final_elev_rotation_radians = (np.pi / 180.0) * final_elev_rotation_angle.type(self.tensor_type)

        origxpos = torch.linspace(0, width - 1, width).repeat(height, 1).repeat(depth, 1, 1).type(
            self.tensor_type) - 2.0 * width / 4.0 + 0.5
        origypos = torch.linspace(0, height - 1, height).repeat(width, 1).permute(1, 0).repeat(depth, 1, 1).type(
            self.tensor_type) - 2.0 * height / 4.0 + 0.5
        origzpos = torch.linspace(0, depth - 1, depth).repeat(height, 1).repeat(width, 1, 1).permute(2, 1, 0).type(
            self.tensor_type) - 2.0 * depth / 4.0 + 0.5

        num_batch = crnt_transform_mode.shape[0]
        origxpos = origxpos.repeat(num_batch, 1, 1, 1)
        origypos = origypos.repeat(num_batch, 1, 1, 1)
        origzpos = origzpos.repeat(num_batch, 1, 1, 1)

        rotxpos = origxpos
        rotypos = origypos
        rotzpos = origzpos

        xpos = rotxpos
        ypos = rotypos
        zpos = rotzpos

        init_elev_cos_rad = torch.cos(
            -init_elev_rotation_radians).reshape(
            num_batch, 1).repeat(1, depth * height * width).reshape(num_batch, depth, height, width)

        init_elev_sin_rad = torch.sin(-init_elev_rotation_radians).reshape(num_batch, 1).repeat(1,
                                                                                                depth * height * width).reshape(
            num_batch, depth, height, width)
        rotypos = torch.mul(init_elev_cos_rad, ypos) + \
                  torch.mul(-init_elev_sin_rad, zpos)
        rotzpos = torch.mul(init_elev_sin_rad, ypos) + \
                  torch.mul(init_elev_cos_rad, zpos)
        ypos = rotypos
        zpos = rotzpos

        cos_rad = torch.cos(
            rotation_radians).reshape(
            num_batch, 1
        ).repeat(
            1, depth * height * width
        ).reshape(
            num_batch, depth, height, width
        )

        sin_rad = torch.sin(
            rotation_radians
        ).reshape(num_batch, 1).repeat(1, depth * height * width).reshape(
            num_batch, depth, height, width
        )

        rotxpos = torch.mul(cos_rad, xpos) + \
                  torch.mul(sin_rad, zpos)

        rotzpos = torch.mul(-sin_rad, xpos) + \
                  torch.mul(cos_rad, zpos)

        zpos = rotzpos

        elev_cos_rad = torch.cos(
            final_elev_rotation_radians
        ).reshape(
            num_batch, 1
        ).repeat(
            1, depth * height * width
        ).reshape(
            num_batch, depth, height, width
        )

        elev_sin_rad = torch.sin(
            final_elev_rotation_radians
        ).reshape(
            num_batch, 1
        ).repeat(
            1, depth * height * width
        ).reshape(
            num_batch, depth, height, width
        )

        rotypos = torch.mul(elev_cos_rad, ypos) + \
                  torch.mul(-elev_sin_rad, zpos)
        rotzpos = torch.mul(elev_sin_rad, ypos) + \
                  torch.mul(elev_cos_rad, zpos)

        flow_field_x = (rotxpos - origxpos).reshape(
            num_batch, 1, encoded_3d_vol.shape[2],
            encoded_3d_vol.shape[3], encoded_3d_vol.shape[4]
        )
        flow_field_y = (rotypos - origypos).reshape(
            num_batch, 1, encoded_3d_vol.shape[2],
            encoded_3d_vol.shape[3], encoded_3d_vol.shape[4]
        )
        flow_field_z = (rotzpos - origzpos).reshape(
            num_batch, 1, encoded_3d_vol.shape[2],
            encoded_3d_vol.shape[3], encoded_3d_vol.shape[4]
        )

        flow_field_x = flow_field_x.view(-1).type(self.tensor_type)
        flow_field_y = flow_field_y.view(-1).type(self.tensor_type)
        flow_field_z = flow_field_z.view(-1).type(self.tensor_type)

        return flow_field_x, flow_field_y, flow_field_z

    def forward(self, num_inputs_to_use, data):
        final_tensor = None
        final_seg_tensor = None
        gen_src_seg2d_image = []
        for input_idx in range(num_inputs_to_use):
            encoded, gen_src_seg2d, seg_encoded = self.encode(input_idx, data)
            gen_src_seg2d_image.append(gen_src_seg2d)
            final_tensor = encoded if final_tensor is None else final_tensor + encoded
            if self.args.use_seg3d_proxy:
                final_seg_tensor = seg_encoded if final_seg_tensor is None else final_seg_tensor + seg_encoded
        final_tensor = final_tensor / num_inputs_to_use

        if self.args.use_seg3d_proxy:
            final_seg_tensor = final_seg_tensor / num_inputs_to_use

        gen_tgt_rgb_image = self.decode(final_tensor)

        if self.args.use_seg3d_proxy:
            gen_tgt_seg3d_image = self.decode_segmentation(final_seg_tensor)
            gen_src_seg3d_image = []
            for input_idx in range(num_inputs_to_use):
                gen_src_seg3d_image.append(self.decode_input_segmentation(input_idx, data, final_seg_tensor))
        else:
            gen_src_seg3d_image = None
            gen_tgt_seg3d_image = None
            final_seg_tensor = None

        return gen_tgt_rgb_image, gen_src_seg3d_image, gen_tgt_seg3d_image, gen_src_seg2d_image, final_seg_tensor

    def encode(self, input_idx, data):
        src_rgb = data['src_rgb_image'][input_idx]
        src_seg = data['src_seg_image'][input_idx]

        src_azim_transform_mode = data['src_azim_transform_mode'][input_idx]
        src_elev_transform_mode = data['src_elev_transform_mode'][input_idx]

        tgt_azim_transform_mode = data['tgt_azim_transform_mode'][0]
        tgt_elev_transform_mode = data['tgt_elev_transform_mode'][0]

        crnt_transform_mode = tgt_azim_transform_mode - src_azim_transform_mode

        x = src_rgb
        if 0 < self.args.num_input_convs:
            x = self.in_conv(x)

        x1 = self.conv1_2d_encode(x)
        x2 = self.conv2_2d_encode(x1)
        x3 = self.conv3_2d_encode(x2)
        x4 = self.conv4_2d_encode(x3)

        skip1 = x1
        skip2 = x2
        skip3 = x3

        x_out = self.upconv4_2d_encode(x4)
        x_out = torch.cat((x_out, skip3), 1)
        x_out = self.iconv4_2d_encode(x_out)

        x_out = self.upconv3_2d_encode(x_out)
        x_out = torch.cat((x_out, skip2), 1)
        x_out = self.iconv3_2d_encode(x_out)

        x_out = self.upconv2_2d_encode(x_out)
        x_out = torch.cat((x_out, skip1), 1)
        x_out = self.iconv2_2d_encode(x_out)

        if self.src_seg2d is not None:
            src_seg2d_output = self.src_seg2d(x_out)
            upsample_src_seg2d_output = self.seg_inv_transform(src_seg2d_output)
        else:
            upsample_src_seg2d_output = None

        depth = self.vol_dim
        height = x_out.shape[2]
        width = x_out.shape[3]

        x_out = x_out.view(x_out.shape[0],
                           int(x_out.shape[1] / self.vol_dim), self.vol_dim,
                           x_out.shape[2], x_out.shape[3])

        x_out = self.conv1_3d_encode(x_out)
        encoded_3d_vol = self.conv2_3d_encode(x_out)

        if self.args.use_seg3d_proxy:
            seg_out = self.seg_conv_3d(encoded_3d_vol)
            seg_out = seg_out.squeeze(1)
            seg_out = src_seg2d_output * seg_out
            seg_encoded_3d_vol = seg_out.unsqueeze(1)

        flow_field_x, flow_field_y, flow_field_z = self.get_flow_fields(crnt_transform_mode,
                                                                        tgt_elev_transform_mode,
                                                                        src_elev_transform_mode,
                                                                        width, height, depth, encoded_3d_vol)

        transformed_output = apply_volume_transform(encoded_3d_vol, flow_field_x, flow_field_y, flow_field_z,
                                                    tensor_type=self.tensor_type)
        seg_transformed_output = None
        if self.args.use_seg3d_proxy:
            seg_transformed_output = apply_volume_transform(seg_encoded_3d_vol, flow_field_x, flow_field_y, flow_field_z,
                                                            tensor_type=self.tensor_type)

        return transformed_output, upsample_src_seg2d_output, seg_transformed_output

    def decode_segmentation(self, final_seg_transformed_output):
        x_out = final_seg_transformed_output.squeeze(1)
        x_out = self.seg_conv_3d_combine(x_out)

        tgt_seg_list = []
        tgt_seg_list.append(x_out)
        return tgt_seg_list

    def decode_input_segmentation(self, input_idx, data, final_seg_transformed_output):
        src_azim_transform_mode = data['src_azim_transform_mode'][input_idx]
        src_elev_transform_mode = data['src_elev_transform_mode'][input_idx]

        tgt_azim_transform_mode = data['tgt_azim_transform_mode'][0]
        tgt_elev_transform_mode = data['tgt_elev_transform_mode'][0]

        crnt_transform_mode = tgt_azim_transform_mode - src_azim_transform_mode

        depth = self.vol_dim
        height = final_seg_transformed_output.shape[2]
        width = final_seg_transformed_output.shape[3]
        flow_field_x, flow_field_y, flow_field_z = self.get_flow_fields(-crnt_transform_mode,
                                                                        src_elev_transform_mode,
                                                                        tgt_elev_transform_mode,
                                                                        width, height, depth,
                                                                        final_seg_transformed_output)

        seg_transformed_output = apply_volume_transform(final_seg_transformed_output, flow_field_x,
                                                        flow_field_y, flow_field_z,
                                                        tensor_type=self.tensor_type)

        return self.seg_conv_3d_combine(seg_transformed_output.squeeze(1))

    def decode(self, final_transformed_output):
        x_out = self.conv1_3d_decode(final_transformed_output)
        x_out = self.conv2_3d_decode(x_out)

        input_2d = x_out.contiguous().view(x_out.shape[0],
                                           x_out.shape[1] * self.vol_dim,
                                           x_out.shape[3], x_out.shape[4])

        x2 = self.conv2_2d_decode(input_2d)
        x3 = self.conv3_2d_decode(x2)
        x4 = self.conv4_2d_decode(x3)

        skip1 = input_2d
        skip2 = x2
        skip3 = x3

        x_out = self.upconv4_2d_decode(x4)
        x_out = torch.cat((x_out, skip3), 1)
        x_out = self.iconv4_2d_decode(x_out)

        x_out = self.upconv3_2d_decode(x_out)
        x_out = torch.cat((x_out, skip2), 1)
        x_out = self.iconv3_2d_decode(x_out)

        x_out = self.upconv2_2d_decode(x_out)
        x_out = torch.cat((x_out, skip1), 1)
        x_out = self.iconv2_2d_decode(x_out)

        x_out = self.upconv1_2d_decode(x_out)
        x_out = self.iconv1_2d_decode(x_out)

        if 0 < self.args.num_output_deconvs:
            x_out = self.deconv(x_out)

        tgt_img = self.output_2d_decode(x_out)
        return tgt_img
