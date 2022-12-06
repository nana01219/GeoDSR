import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from .GASA import GASA
from models.edsr import ResBlock
# from models.Involution import involution

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    return ret

class Feature_modulator(nn.Module):

    def __init__(self, in_dim, out_dim, act = True):
        super().__init__()

        self.act = nn.LeakyReLU()
        self.head    = nn.Sequential(
            nn.Conv2d(in_channels = in_dim, out_channels = out_dim, kernel_size= 3, padding=1),
            self.act
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size= 1),
            self.act
        )
        
        self.body1 = ResBlock(conv=default_conv, n_feats=out_dim, kernel_size=3)
        self.body2 = ResBlock(conv=default_conv, n_feats=out_dim, kernel_size=3)

        if act:
            self.imnet   = nn.Sequential(
                nn.Conv2d(in_channels = out_dim, 
                            out_channels = out_dim, kernel_size= 3, padding=1),
                nn.LeakyReLU(),
                )
        else:
            self.imnet   = nn.Sequential(
                nn.Conv2d(in_channels = out_dim, 
                            out_channels = out_dim, kernel_size= 3, padding=1)
                )

        
    def gen_coord(self, in_shape, output_size):

        self.image_size =output_size
        self.coord      = make_coord(output_size,flatten=False) \
                            .expand(in_shape[0],output_size[0],output_size[1],2).flip(-1)      
        self.coord      = self.coord.cuda()


    def forward(self, feat, guide_hr, cell):
        q_feat = F.grid_sample(
            feat, self.coord, mode='bilinear', align_corners=False)
        guide_hr = F.grid_sample(
            guide_hr, self.coord, mode='bilinear', align_corners=False)
        # q_feat = self.conv1x1(q_feat)

        # qian or hou ?
        # q_feat = torch.cat([q_feat, guide_hr], dim=1)
        # guide_hr = self.conv1x1(guide_hr)
        q_feat = q_feat*guide_hr
        q_feat = self.conv1x1(q_feat)
        q_feat  = self.head(q_feat)

        b_out1 = self.body1(q_feat)
        b_out2 = self.body2(b_out1)
        q_feat = q_feat + b_out2

        q_feat = self.imnet(q_feat)
        return q_feat

class GASA_model(nn.Module):

    def __init__(self, args, feat_dim=128, guide_dim=128, block_num = 16, mlp_dim=[1024,512,256,128]):
        super().__init__()
        self.args = args
        # self.feat_dim = feat_dim
        # self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim

        self.image_encoder = GASA(in_dim=3, dim=128, num_groups=guide_dim, num_blocks=block_num, use_CA=True, group_A="ESA")
        self.depth_encoder = GASA(in_dim=1, dim=128, num_groups=feat_dim, num_blocks=block_num, use_CA=True, group_A="ESA")

        self.upsample1 = Feature_modulator(128, 128)
        self.upsample2 = Feature_modulator(128, 64)

        self.decoder = nn.Conv2d(in_channels = 64, out_channels =1, kernel_size=3, stride=1, padding = 1)
        # print("---------------> model: GASA")
        

    def gen_DSF_coord(self, coord, in_size, out_size, m_scal_factor, N):
        m_size_h = int(in_size[0] * m_scal_factor[0])
        m_size_w = int(in_size[1] * m_scal_factor[1])

        self.upsample1.gen_coord((N,\
            1, in_size[0], in_size[1]), (m_size_h, m_size_w))
        self.upsample2.coord = coord.flip(-1)
        return [m_size_h, m_size_w]


    def forward(self, data):
        hr_image, lr_depth, coord, lr_depth_up, field = data['hr_image'], data['lr_depth'], data['hr_coord'], data['lr_depth_up'], data["field"]

        # cal distance/geo information for warped cases
        scale = field.mean(dim=1).unsqueeze(-1)
        
        hr_guide = self.image_encoder(hr_image, scale)
        feat = self.depth_encoder(lr_depth, scale)

        N, in_h, in_w = feat.shape[0], feat.shape[-2], feat.shape[-1]
        out_h = int(coord.shape[1])
        out_w = int(coord.shape[2])
        sum_scal_factor_h = out_h/in_h
        sum_scal_factor_w = out_w/in_w
        scal_factor_h = sum_scal_factor_h ** 0.5
        scal_factor_w = sum_scal_factor_w ** 0.5
        m_size = self.gen_DSF_coord(coord, (in_h,in_w), (out_h, out_w), (scal_factor_h,scal_factor_w), N)

        pred = self.upsample1(feat, hr_guide, True)
        pred = self.upsample2(pred, hr_guide, True)
        pred = self.decoder(pred)

        res = lr_depth_up + pred

        return res


