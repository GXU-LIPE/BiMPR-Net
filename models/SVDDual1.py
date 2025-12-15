from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import gather_operation as gather_points
import time
from models.model_utils import *
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from encoder_image.resnet import ResNet


class cross_transformer(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

        self.trans_conv = nn.Conv1d(d_model_out, d_model_out, 1)
        self.after_norm = nn.BatchNorm1d(d_model_out)
        self.act = nn.ReLU()

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)
        # x = src1
        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1, key=src2, value=src2)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = src1.permute(1, 2, 0)

        return src1


class PCT_refine(nn.Module):
    def __init__(self, channel=128, rate=1):
        super(PCT_refine, self).__init__()
        self.ratio = rate
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = cross_transformer(channel*2, 768)
        self.sa2 = cross_transformer(768, 512)
        self.sa3 = cross_transformer(512, channel*rate)
        self.cross1 = cross_attention(768, 768, dropout=0.0, nhead=8)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*rate, channel*rate, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.cpe1 = ConvPos(dim_q=768, dim_content=1)

    def forward(self, local_feat, feat, Feat):
        batch_size, _, N = Feat.size()
        # y = self.conv_x1(self.relu(self.conv_x(coarse)))  # B, C, N
        # feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        # y0 = torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)

        y1 = self.sa1(Feat, Feat)
        x = self.cpe1(q=y1)
        y4 = self.cross1(local_feat, x)

        y2 = self.sa2(y4, y4)
        y3 = self.sa3(y2, y2)
        # y3 = self.conv_ps(y3)
        # y3 = self.conv_ps(y3).reshape(batch_size, -1, N*self.ratio)

        # y_up = feat.repeat(1, 1, self.ratio)
        # y_cat = torch.cat([y3, y_up], dim=1)
        # y4 = self.conv_delta(y_cat)
        #
        # x = self.conv_out(self.relu(self.conv_out1(y4)))

        return y1, y3


class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=256):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points


class SDG(nn.Module):
    def __init__(self, channel=128, ratio=1, hidden_dim=512):
        super(SDG, self).__init__()
        self.channel = channel
        self.hidden = hidden_dim

        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = self_attention(channel*2, hidden_dim, dropout=0.0, nhead=8)
        self.cross1 = cross_attention(hidden_dim, hidden_dim, dropout=0.0, nhead=8)

        self.decoder1 = SDG_Decoder(hidden_dim, channel, ratio)
        self.decoder2 = SDG_Decoder(hidden_dim, channel, ratio)

        self.relu = nn.GELU()
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_delta = nn.Conv1d(channel, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio*2, channel*ratio, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.mlpp = MLP_CONV(in_channel=256, layer_dims=[256, hidden_dim])
        self.sigma_d = 0.2
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.cd_distance = chamfer_3DDist()
        self.refine = PCT_refine(rate=4)
        self.cpe = ConvPosEnc(dim_q=768, dim_content=1)

    def forward(self, local_feat, coarse, f_g, partial):
        batch_size, _, N = coarse.size()
        f = self.conv_x1(self.relu(self.conv_x(coarse)))
        f_g = self.conv_1(self.relu(self.conv_11(f_g)))
        F = torch.cat([f, f_g.repeat(1, 1, f.shape[-1])], dim=1)

        # Structure Analysis
        local_feat = self.mlpp(local_feat)
        feat_corse, feat_fine = self.refine(local_feat, f, F)
        # half_cd = self.cd_distance(coarse.transpose(1, 2).contiguous(), partial.transpose(1, 2).contiguous())[0] / self.sigma_d
        # embd = self.embedding(half_cd).reshape(batch_size, self.hidden, -1).permute(2, 0, 1)
        # F_Q = self.sa1(F, embd)  # B 768 512
        # F_Q_ = self.decoder1(F_Q)  # 16 512 512

        # Similarity Alignment
        x = self.cpe(q=feat_corse)
        F_H = self.cross1(x, local_feat)
        F_H_ = self.decoder2(F_H)

        F_L = self.conv_delta(self.conv_ps(torch.cat([feat_fine, F_H_], 1)).reshape(batch_size, -1, N*self.ratio))
        O_L = self.conv_out(self.relu(self.conv_out1(F_L)))
        fine = coarse.repeat(1, 1, self.ratio) + O_L

        return fine


class SVFNet(nn.Module):
    def __init__(self):
        super(SVFNet, self).__init__()
        self.channel = 64
        self.point_feature_extractor = FeatureExtractor()
        self.view_distance = 1.5
        self.relu = nn.GELU()
        self.sa = self_attention(self.channel*8, self.channel*8, dropout=0.0)
        self.viewattn = self_attention(128+256, 256)
        self.attn = self_attention(128+128, 128)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(512+self.channel*4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(512, self.channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(512+self.channel, self.channel*8, kernel_size=1)

        img_layers, in_features = self.get_img_layers('resnet18', feat_size=16)
        self.img_feature_extractor = nn.Sequential(*img_layers)
        self.posmlp = MLP_CONV(3, [64, 256])
        self.posmlp_1 = MLP_CONV(3, [64, 128])
        self.im_encoder = ResNet()

        self.cross_attn1 = nn.MultiheadAttention(128, 4, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(128)

        self.self_attn1 = nn.MultiheadAttention(128, 4, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(128)

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        from models.resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        return img_layers, in_features

    def forward(self, points, depth, image):
        batch_size, _, N = points.size()
        f_v = self.img_feature_extractor(depth).view(batch_size, 1, -1).transpose(1, 2).contiguous()

        f_i = self.im_encoder(image).view(batch_size, 1, -1).transpose(1, 2).contiguous()

        f_p = self.point_feature_extractor(points)

        # View Augment
        single_view = torch.tensor([0, 0, -self.view_distance], dtype=torch.float32).view(1, 3, 1).expand(batch_size, 3, 1).to(depth.device)
        single_view_feature = self.posmlp(single_view).permute(2, 0, 1)

        single_view_feature_1 = self.posmlp_1(single_view).permute(2, 0, 1)
        f_v = self.attn(torch.cat([f_v, f_i], 1), single_view_feature_1)

        f_v_ = self.viewattn(torch.cat([f_v, f_p], 1))
        f_v_ = F.adaptive_max_pool1d(f_v_, 1)
        f_g = torch.cat([f_p, f_v_], 1)

        x = self.relu(self.ps(f_g))
        x = self.relu(self.ps_refuse(torch.cat([x, f_g.repeat(1, 1, x.size(2))], 1)))
        x2_d = (self.sa(x)).reshape(batch_size, self.channel*4, N//8)
        coarse = self.conv_out(self.relu(self.conv_out1(torch.cat([x2_d, f_g.repeat(1, 1, x2_d.size(2))], 1))))

        return f_g, coarse


class local_encoder(nn.Module):
    def __init__(self):
        super(local_encoder, self).__init__()
        self.gcn_1 = EdgeConv(3, 64, 16)
        self.gcn_2 = EdgeConv(64, 256, 8)
        self.local_number = 512

    def forward(self, input):
        x1 = self.gcn_1(input)
        idx = furthest_point_sample(input.transpose(1, 2).contiguous(), self.local_number)
        x1 = gather_points(x1, idx)
        x2 = self.gcn_2(x1)

        return x2


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.encoder = SVFNet()
        self.localencoder = local_encoder()
        self.merge_points = 512
        self.refine1 = SDG(ratio=4, hidden_dim=768)
        self.refine2 = SDG(ratio=2, hidden_dim=512)

    def forward(self, partial, depth, image):
        partial = partial.transpose(1, 2).contiguous()
        feat_g, coarse = self.encoder(partial, depth, image)
        local_feat = self.localencoder(partial)

        coarse_merge = torch.cat([partial, coarse], dim=2)
        coarse_merge = gather_points(coarse_merge, furthest_point_sample(coarse_merge.transpose(1, 2).contiguous(), self.merge_points))

        fine1 = self.refine1(local_feat, coarse_merge, feat_g, partial)
        # fine2 = self.refine2(local_feat, fine1, feat_g, partial)

        # return coarse.transpose(1, 2).contiguous(), fine1.transpose(1, 2).contiguous(), fine2.transpose(1, 2).contiguous()
        return coarse.transpose(1, 2).contiguous(), fine1.transpose(1, 2).contiguous()


# Local Convolutional Position Encoding
class ConvPosEnc(nn.Module):
    def __init__(self, dim_q, dim_content, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj_q = nn.Conv1d(
            in_channels=dim_q,
            out_channels=dim_q,
            kernel_size=k,
            stride=1,
            padding=k//2,
            groups=dim_q
        )

        self.proj_content = nn.Conv1d(
            in_channels=dim_content,
            out_channels=dim_content,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=dim_content
        )

    def forward(self, q):
        # q = q.permute(0, 2, 1)
        q = self.proj_q(q) + q
        # q = q.permute(0, 2, 1)

        # # B,C,H,W = content.shape
        # content = content.permute(0, 2, 1)
        # content = self.proj_content(content) + content
        # content = content.permute(0, 2, 1)

        return q


class ConvPos(nn.Module):
    def __init__(self, dim_q, dim_content, k=3):
        super(ConvPos, self).__init__()
        self.proj_q = nn.Conv1d(
            in_channels=dim_q,
            out_channels=dim_q,
            kernel_size=k,
            stride=1,
            padding=k//2,
            groups=1
        )

        self.proj_content = nn.Conv1d(
            in_channels=dim_content,
            out_channels=dim_content,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=dim_content
        )

    def forward(self, q):
        # q = q.permute(0, 2, 1)
        q = self.proj_q(q) + q
        # q = q.permute(0, 2, 1)

        # # B,C,H,W = content.shape
        # content = content.permute(0, 2, 1)
        # content = self.proj_content(content) + content
        # content = content.permute(0, 2, 1)

        return q

