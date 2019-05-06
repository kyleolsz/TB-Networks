from __future__ import absolute_import, division, print_function
import torch
from torch.nn.functional import pad


def apply_volume_transform(input_volume, x_offset, y_offset, z_offset,
                           tensor_type='torch.cuda.FloatTensor'):
    num_batch, num_channels, depth, height, width = input_volume.size()

    im_flat = input_volume.permute(1, 0, 2, 3, 4).contiguous().view(num_channels, -1)

    x = torch.linspace(0, width - 1, width).repeat(depth, height, 1).type(tensor_type)
    y = torch.linspace(0, height - 1, height).repeat(depth, width, 1).permute(0, 2, 1).type(tensor_type)
    z = torch.linspace(0, depth - 1, depth).repeat(height, width, 1).permute(2, 0, 1).type(tensor_type)

    x = x.contiguous().view(-1).repeat(1, num_batch)
    y = y.contiguous().view(-1).repeat(1, num_batch)
    z = z.contiguous().view(-1).repeat(1, num_batch)

    x = x + x_offset.contiguous().view(-1)
    y = y + y_offset.contiguous().view(-1)
    z = z + z_offset.contiguous().view(-1)

    x = torch.clamp(x, 0.0, width - 1)
    y = torch.clamp(y, 0.0, height - 1)
    z = torch.clamp(z, 0.0, depth - 1)

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1
    z0 = torch.floor(z)
    z1 = z0 + 1

    x1 = x1.clamp(max=(width - 1))
    y1 = y1.clamp(max=(height - 1))
    z1 = z1.clamp(max=(depth - 1))

    dim3 = width
    dim2 = width * height
    dim1 = width * height * depth

    base = dim1 * torch.arange(num_batch).type(tensor_type)
    base = base.view(-1, 1).repeat(1, depth * height * width).view(-1)

    base_z0 = base + z0 * dim2
    base_z1 = base + z1 * dim2

    base_y0z0 = base_z0 + y0 * dim3
    base_y0z1 = base_z1 + y0 * dim3

    base_y1z0 = base_z0 + y1 * dim3
    base_y1z1 = base_z1 + y1 * dim3

    idx_lun = base_y0z0 + x0
    idx_luf = base_y0z1 + x0

    idx_run = base_y0z0 + x1
    idx_ruf = base_y0z1 + x1

    idx_ldn = base_y1z0 + x0
    idx_ldf = base_y1z1 + x0

    idx_rdn = base_y1z0 + x1
    idx_rdf = base_y1z1 + x1

    pix_lun = im_flat.gather(1, idx_lun.repeat(num_channels, 1).long())
    pix_luf = im_flat.gather(1, idx_luf.repeat(num_channels, 1).long())

    pix_run = im_flat.gather(1, idx_run.repeat(num_channels, 1).long())
    pix_ruf = im_flat.gather(1, idx_ruf.repeat(num_channels, 1).long())

    pix_ldn = im_flat.gather(1, idx_ldn.repeat(num_channels, 1).long())
    pix_ldf = im_flat.gather(1, idx_ldf.repeat(num_channels, 1).long())

    pix_rdn = im_flat.gather(1, idx_rdn.repeat(num_channels, 1).long())
    pix_rdf = im_flat.gather(1, idx_rdf.repeat(num_channels, 1).long())

    length_l = (x1 - x)
    length_r = (x - x0)

    length_u = (y1 - y)
    length_d = (y - y0)

    length_n = (z1 - z)
    length_f = (z - z0)

    weight_lun = length_l * length_u * length_n
    weight_luf = length_l * length_u * length_f

    weight_run = length_r * length_u * length_n
    weight_ruf = length_r * length_u * length_f

    weight_ldn = length_l * length_d * length_n
    weight_ldf = length_l * length_d * length_f

    weight_rdn = length_r * length_d * length_n
    weight_rdf = length_r * length_d * length_f

    output = weight_lun * pix_lun + weight_luf * pix_luf + \
             weight_run * pix_run + weight_ruf * pix_ruf + \
             weight_ldn * pix_ldn + weight_ldf * pix_ldf + \
             weight_rdn * pix_rdn + weight_rdf * pix_rdf

    return output.view(num_channels, num_batch, depth, height, width).permute(1, 0, 2, 3, 4)
