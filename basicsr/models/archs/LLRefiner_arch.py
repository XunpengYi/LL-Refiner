import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter
from timm.layers import DropPath


class LL_Refiner(nn.Module):
    def __init__(self,
                 coarse_dim=32,
                 en_feature_num=48,
                 en_inter_num=32,
                 de_feature_num=64,
                 de_inter_num=32,
                 sam_number=2,
                 ):
        super(LL_Refiner, self).__init__()
        self.coarse_estimation = Coarse_Stage_Estimator(dim=coarse_dim)
        self.fine_estimation = Fine_Stage_Estimator(en_feature_num=en_feature_num, en_inter_num=en_inter_num,
                                                    de_feature_num=de_feature_num,
                                                    de_inter_num=de_inter_num, sam_number=sam_number)
        self._initialize_weights()

    def forward(self, x):
        x_coarse = self.coarse_estimation(x)
        out_0, out_1, out_2 = self.fine_estimation(x, x_coarse)
        out_0 = out_0 + x
        return out_0, out_1, out_2, x_coarse

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class Fine_Stage_Estimator(nn.Module):
    def __init__(self,
                 en_feature_num=64,
                 en_inter_num=32,
                 de_feature_num=64,
                 de_inter_num=32,
                 sam_number=2,
                 ):
        super(Fine_Stage_Estimator, self).__init__()
        self.encoder = Encoder(feature_num=en_feature_num, inter_num=en_inter_num, sam_number=sam_number)
        self.decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num,
                               sam_number=sam_number)

    def forward(self, x, x_coarse):
        y_0, y_1, y_2 = self.encoder(x, x_coarse)
        out_0, out_1, out_2 = self.decoder(y_0, y_1, y_2)
        return out_0, out_1, out_2


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number):
        super(Decoder, self).__init__()
        self.preconv_2 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, sam_number, large_block=True)

        self.preconv_1 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_0 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_0 = Decoder_Level_Full(feature_num)

    def forward(self, y_0, y_1, y_2):
        x_2 = self.preconv_2(y_2)
        out_2, feat_2 = self.decoder_2(x_2)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1, feat_1 = self.decoder_1(x_1)

        x_0 = torch.cat([y_0, feat_1], dim=1)
        x_0 = self.preconv_0(x_0)
        out_0 = self.decoder_0(x_0)

        return out_0, out_1, out_2


class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number, num_heads=[2, 4]):
        super(Encoder, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, feature_num, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU()
        )

        self.pixelshuffle_up = PixelShuffleUpsample(3, 2)

        self.encoder_1 = Encoder_Level_Full(feature_num, level=1)
        self.encoder_2 = Encoder_Level_Fusion(2 * feature_num, inter_num, level=2, sam_number=sam_number,
                                              num_heads=num_heads[0])
        self.encoder_3 = Encoder_Level_Fusion(4 * feature_num, inter_num, level=3, sam_number=sam_number,
                                              num_heads=num_heads[1])

    def forward(self, x, x_coarse_down_4):
        x = self.conv_first(x)
        out_feature_1, down_feature_1 = self.encoder_1(x)
        x_coarse_down_2 = self.pixelshuffle_up(x_coarse_down_4)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1, x_coarse_down_2)
        out_feature_3 = self.encoder_3(down_feature_2, x_coarse_down_4)

        return out_feature_1, out_feature_2, out_feature_3


class Encoder_Level_Full(nn.Module):
    def __init__(self, feature_num, level):
        super(Encoder_Level_Full, self).__init__()
        self.convblock = ConvBlock(feature_num)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.GELU()
            )
        self.level = level

    def forward(self, x):
        out_feature = self.convblock(x)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number):
        super(Encoder_Level, self).__init__()
        self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.GELU()
            )
        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)
        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class Refinement_Attention(nn.Module):
    def __init__(self, dim, out_dim, level, num_heads, bias=False):
        super(Refinement_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * (dim // num_heads) ** -0.5)

        if level == 3:
            attn_dim = dim // 2
        else:
            attn_dim = dim

        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, attn_dim, kernel_size=1, bias=bias),
            nn.Conv2d(attn_dim, attn_dim, kernel_size=3, padding=1, groups=attn_dim, bias=bias)
        )
        self.kv_proj = nn.Sequential(
            nn.Conv2d(dim, attn_dim * 2, kernel_size=1, bias=bias),
            nn.Conv2d(attn_dim * 2, attn_dim * 2, kernel_size=3, padding=1, groups=attn_dim * 2, bias=bias)
        )

        self.project_out = nn.Conv2d(attn_dim, out_dim, kernel_size=1, bias=bias)

    def forward(self, x, x_coarse):
        b, _, h, w = x.shape

        q = self.q_proj(x_coarse)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v) + v  # add residual
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Adaptive_Refinement_Module(nn.Module):
    def __init__(self, inp_feature_num, out_feature_num, level, num_heads):
        super().__init__()
        self.atten = Refinement_Attention(dim=inp_feature_num, out_dim=out_feature_num, level=level,
                                          num_heads=num_heads)

    def forward(self, x, x_coarse):
        x = self.atten(x, x_coarse)
        return x


class Encoder_Level_Fusion(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number, num_heads):
        super(Encoder_Level_Fusion, self).__init__()
        self.input_semi = nn.Conv2d(3, feature_num, kernel_size=3, stride=1, padding=1, bias=False)
        self.adaptive_refinement_module = Adaptive_Refinement_Module(inp_feature_num=feature_num,
                                                                     out_feature_num=feature_num, level=level,
                                                                     num_heads=num_heads)

        self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.GELU()
            )
        self.level = level

    def forward(self, x, x_coarse):
        x_coarse = self.input_semi(x_coarse)
        x = self.adaptive_refinement_module(x, x_coarse)

        out_feature = self.rdb(x)
        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class Decoder_Level_Full(nn.Module):
    def __init__(self, feature_num):
        super(Decoder_Level_Full, self).__init__()
        self.convblock = ConvBlock(feature_num)
        self.conv = conv(in_channel=feature_num, out_channel=3, kernel_size=1)

    def forward(self, x):
        x = self.convblock(x)

        feature_out = F.gelu(x)
        out = self.conv(feature_out)
        return out


class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number, large_block=False):
        super(Decoder_Level, self).__init__()
        self.rdb = RDB(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            if large_block:
                sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            else:
                sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)
        self.conv = conv(in_channel=feature_num, out_channel=3, kernel_size=1)
        self.pixelshuffle = PixelShuffleUpsample(feature_num, 2)

    def forward(self, x, feat=True):
        x = self.rdb(x)
        for sam_block in self.sam_blocks:
            x = sam_block(x)

        feature_out = F.gelu(x)
        out = self.conv(feature_out)

        feature = self.pixelshuffle(x)

        if feat:
            return out, feature
        else:
            return out


class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, bias=False):
        super().__init__()
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 1x1 Conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)  # depthwise conv
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 1x1 Conv
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.pwconv1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = input + self.drop_path(x)
        return x


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class SAM(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(SAM, self).__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion = CSAF(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y


class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        self.in_chnls = in_chnls

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_chnls, in_chnls // ratio, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_chnls // ratio, in_chnls, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x0, x2, x4):
        s0 = self.squeeze(x0)
        s2 = self.squeeze(x2)
        s4 = self.squeeze(x4)
        s = torch.cat([s0, s2, s4], dim=1)
        weights = self.channel_mlp(s)
        w0, w2, w4 = torch.chunk(weights, 3, dim=1)
        out = x0 * w0 + x2 * w2 + x4 * w4
        return out


class RDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(RDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.GELU()
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward_Restormer(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_Restormer, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * (dim ** -0.5))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        #######################################
        self.illumination_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        illumination_map = self.illumination_attn(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.temperature
        attn = attn.softmax(dim=-1)

        ####################
        out = (attn @ v) + v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        #################
        out = out * illumination_map
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_Restormer(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed_Restormer(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed_Restormer, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
class Coarse_Stage_Estimator(nn.Module):
    def __init__(self,
                 inp_channels=48,
                 out_channels=3,
                 dim=32,
                 num_blocks=[1, 2],
                 heads=[1, 2],
                 ffn_expansion_factor=1.5,
                 bias=False,
                 LayerNorm_type='WithBias'  ## Other option 'BiasFree'
                 ):
        super(Coarse_Stage_Estimator, self).__init__()

        self.patch_embed = OverlapPatchEmbed_Restormer(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_img_in = F.pixel_unshuffle(inp_img, 4)
        inp_enc_level1 = self.patch_embed(inp_img_in)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_dec_level2 = out_enc_level2
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        inp_img_down_x4 = F.interpolate(inp_img, scale_factor=0.25, mode='bicubic', align_corners=False)
        out_dec_level1 = self.output(out_dec_level1) + inp_img_down_x4
        return out_dec_level1

###############################################################################################