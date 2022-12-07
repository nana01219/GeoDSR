from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange
import math

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Channel attention

class Simple_CA_layer(nn.Module):
    def __init__(self, channel):
        super(Simple_CA_layer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)
                      
    def forward(self, x):
        return x * self.fc(self.gap(x))

class ECA_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel): 
        super(ECA_layer, self).__init__()

        b = 1
        gamma = 2
        k_size = int(abs(math.log(channel,2)+b)/gamma)
        k_size = k_size if k_size % 2 else k_size+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

##########################################################################
## FFD
class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1  = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2  = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3  = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4  = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid= nn.Sigmoid()
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_     = (self.conv1(x))
        c1      = self.conv2(c1_)
        v_max   = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3      = self.conv3(v_max)
        c3      = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf      = self.conv_f(c1_)
        c4      = self.conv4(c3 + cf)
        m       = self.sigmoid(c4)
        return x * m

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        self.to_style = nn.Linear(1, in_chan)
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, scale):
        b, c, h, w = x.shape

        y = self.to_style(scale)
        w1 = y[:, None, :, None, None]
        # print('w1--', w1.shape)
        w2 = self.weight[None, :, :, :, :]
        # print('w2--',w2.shape)
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)
        # print(x.shape)

        x = x.reshape(-1, self.filters, h, w)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, use_CA):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.use_CA = use_CA
        
        self.conv1 = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(hidden_features*2, dim, kernel_size=1, bias=True)

        if use_CA:
            self.CA = ECA_layer(dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        if self.use_CA:
            x = self.CA(x)
        return x

class FeedForward_restormer(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_restormer, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FeedForward_scale_chunk(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_scale_chunk, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.scale_conv = Conv2DMod(hidden_features, hidden_features, kernel=3)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, scale):
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        x1 = self.dwconv(x1)
        x2 = self.scale_conv(x2, scale)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FeedForward_noCA(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_noCA, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        # self.scale_conv = Conv2DMod(hidden_features, hidden_features, kernel=1)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1= self.project_in(x)
        x1 = self.dwconv(x1)
        # x2 = self.scale_conv(x2, scale)
        x = F.gelu(x1)
        x = self.project_out(x)
        return x

##########################################################################
## Geometric 
class Scale_Conv(nn.Module):
    def __init__(self, dim, bias=True):
        super(Scale_Conv, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.mod = Conv2DMod(dim, dim, kernel=3)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, scale):
        y = self.conv1(x)
        y = self.mod(F.gelu(y), scale)
        y = self.conv2(y)
        return y

class Scale_Conv_vgroups(nn.Module):
    def __init__(self, dim, bias=True):
        super(Scale_Conv_vgroups, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.mod = Conv2DMod(dim, dim, kernel=3, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, scale):
        y = self.conv1(x)
        y = self.mod(F.gelu(y), scale)
        y = self.conv2(y)
        return y

class Conv_ablation(nn.Module):
    def __init__(self, dim, bias=True):
        super(Conv_ablation, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.mod = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, scale):
        y = self.conv1(x)
        y = self.mod(F.gelu(y))
        y = self.conv2(y)
        return y

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
         
##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, LayerNorm_type, use_CA, attn_type, ffd_type="ours"):
        super(TransformerBlock, self).__init__()

        print("TransFormerBlock type -- ", attn_type, ffd_type)     

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if attn_type == "ResFormer":
            pass
            # self.attn = ResOperator(dim)
        elif attn_type == "self-attention":
            self.attn = Attention(dim, num_heads=8)
        else:
            raise Exception("self.attn not implement")

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        if ffd_type == "ours":
            self.ffn = FeedForward(dim, ffn_expansion_factor, use_CA=use_CA)
        elif ffd_type == "Restormer":
            self.ffn = FeedForward_restormer(dim, ffn_expansion_factor, bias=False)
        elif ffd_type == "Restormer_noCA":
            self.ffn = FeedForward_noCA(dim, ffn_expansion_factor, bias=False)
        else:
            raise Exception("self.ffn not implement")

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Scale_TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, LayerNorm_type, use_CA, attn_type, ffd_type):
        super(Scale_TransformerBlock, self).__init__()

        print("ScaleTransFormerBlock -- ", attn_type, ffd_type)     

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if attn_type == "scale_conv":
            self.attn = Scale_Conv(dim)
        elif attn_type == "scale_conv_groups":
            self.attn = Scale_Conv_vgroups(dim)
        elif attn_type == "conv_ablation":
            self.attn = Conv_ablation(dim)
        else:
            raise Exception("self.attn not implement")

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        if ffd_type == "ours":
            self.ffn = FeedForward(dim, ffn_expansion_factor, use_CA=use_CA)
        elif ffd_type == "Restormer":
            self.ffn = FeedForward_restormer(dim, ffn_expansion_factor, bias=False)
        else:
            raise Exception("self.ffn not implement")

    def forward(self, input):
        x, scale = input
        assert scale is not None
        x = x + self.attn(self.norm1(x), scale)
        x = x + self.ffn(self.norm2(x))
        return x, scale

##########################################################################
## Groups

# v1: TransformerBlock--ResFormer + our FFD; use_CA+CA; 
class Scale_Groups(nn.Module):
    def __init__(self, dim, num_blocks, use_CA, group_A, attn_type, ffd_type):
        super(Scale_Groups, self).__init__()
        
        self.body = nn.Sequential(*[Scale_TransformerBlock(dim=dim, ffn_expansion_factor=2.66, LayerNorm_type="WithBias", use_CA=use_CA, attn_type=attn_type, ffd_type=ffd_type) for i in range(num_blocks)])
        self.out_layer = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        
        if group_A == "ECA":
            self.DSF_CA = ECA_layer(dim)
        elif group_A == "ESA":
            esa_channel     = max(dim // 4, 16)
            self.DSF_CA     = ESA(esa_channel, dim)
        elif group_A == "noESA":
            self.DSF_CA = None

        # print("--DASA Groups", num_blocks, group_A)

    def forward(self, input):
        x, scale = input
        # input = (x, scale)
        feat, _ = self.body(input)
        feat = x + self.out_layer(feat)
        if self.DSF_CA is not None:
            feat = self.DSF_CA(feat)
        return feat, scale

class GASA(nn.Module):
    def __init__(self, in_dim, dim, num_groups, num_blocks, use_CA = True, group_A = "ESA", attn_type="scale_conv", ffd_type="Restormer", long_res=False):
        super(GASA, self).__init__()
        self.encoder = nn.Conv2d(in_channels=in_dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.groups = nn.Sequential(*[Scale_Groups(dim=dim, num_blocks=num_blocks, use_CA=use_CA, group_A=group_A, attn_type=attn_type, ffd_type=ffd_type) for i in range(num_groups)])
        self.long_res = long_res
        if self.long_res:
            self.res_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        print("GASA condition --", num_groups, "x", num_blocks, group_A, long_res)

    def forward(self, x, scale):
        feat = self.encoder(x)
        input = (feat, scale)
        x, _ = self.groups(input)
        if self.long_res:
            x = x + feat
            x = self.res_conv(x)
        return x





        



        

