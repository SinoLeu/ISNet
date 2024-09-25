import torch.nn as nn
from spikingjelly.clock_driven import layer

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial

from modules.neuron import MultiStepMultiLIFNeuron

__all__ = ['spikformer']


## MLP
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepMultiLIFNeuron()

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepMultiLIFNeuron()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,N,C = x.shape
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)
        return x
    
class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepMultiLIFNeuron()

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepMultiLIFNeuron()

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepMultiLIFNeuron()
        self.attn_lif = MultiStepMultiLIFNeuron()

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepMultiLIFNeuron()

    def forward(self, x):
        T,B,N,C = x.shape

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class VAMutilSpikingformer(nn.Module):
    def __init__(self,
                 m1_c,
                 m2_c,
                 m2_dim,
                 dim,neuron_dropout=0.3):
        super(VAMutilSpikingformer, self).__init__()
        self.audio_conv1 = nn.Conv2d(m2_c, m1_c, kernel_size=1)
        self.audio_projection = nn.Linear(m2_dim,dim)
        tr_dim = 512
        num_heads = 8
        layers = 4
        ## TODO init a model list 
        self.blocks = nn.ModuleList([
            Block(tr_dim, num_heads) for _ in range(layers)
        ])
        self.c_head = nn.Linear()
        # self.neuron_model = MultiStepMultiLIFNeuron
    def forward(self,x_img,x_audio):
        x_audio = self.audio_projection(self.audio_conv1(x_audio))
        img_audo_tensor = torch.cat((x_img, x_audio),dim=1)
        for block in self.blocks:
            img_audio_tensor = block(img_audio_tensor)
        pass
    # pass

# class VAMutilSpikingResNet(nn.Module):
#     def __init__(self,
#                  m1_c,
#                  m2_c,
#                  m2_dim,
#                  dim,spiking_resnet,neuron_dropout=0.3):
#         super(VAMutilSpikingResNet, self).__init__()
#         self.audio_conv1 = nn.Conv2d(m2_c, m1_c, kernel_size=1)  # 输出尺寸: (16, 512, 512)
#         self.audio_projection = nn.Linear(m2_dim,dim)
#         self.neuron_model = MultiLIFNeuron
#         self.spiking_net = spiking_resnet(neuron=self.neuron_model,neuron_dropout=neuron_dropout,c_in=6)
#     def forward(self,x_img,x_audio):
#         x_audio = self.audio_projection(self.audio_conv1(x_audio))
#         img_audo_tensor = torch.cat((x_img, x_audio),dim=1)
#         return self.spiking_net(img_audo_tensor)
    
class VAMutilSpikingTranformer(nn.Module):
    def __init__(self,m1_c,m2_c,m2_dim,dim,neuron_dropout=0.3):
        super(VAMutilSpikingTranformer, self).__init__()
        
        # pass
    pass

# class SPS(nn.Module):
#     def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
#         super().__init__()
#         self.image_size = [img_size_h, img_size_w]
#         patch_size = to_2tuple(patch_size)
#         self.patch_size = patch_size
#         self.C = in_channels
#         self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
#         self.num_patches = self.H * self.W
#         self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
#         self.proj_bn = nn.BatchNorm2d(embed_dims//8)
#         self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

#         self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
#         self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
#         self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

#         self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
#         self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
#         self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
#         self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

#         self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
#         self.proj_bn3 = nn.BatchNorm2d(embed_dims)
#         self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
#         self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

#         self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
#         self.rpe_bn = nn.BatchNorm2d(embed_dims)
#         self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

#     def forward(self, x):
#         T, B, C, H, W = x.shape
#         x = self.proj_conv(x.flatten(0, 1)) # have some fire value
#         x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
#         x = self.proj_lif(x).flatten(0, 1).contiguous()

#         x = self.proj_conv1(x)
#         x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
#         x = self.proj_lif1(x).flatten(0, 1).contiguous()

#         x = self.proj_conv2(x)
#         x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
#         x = self.proj_lif2(x).flatten(0, 1).contiguous()
#         x = self.maxpool2(x)

#         x = self.proj_conv3(x)
#         x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).contiguous()
#         x = self.proj_lif3(x).flatten(0, 1).contiguous()
#         x = self.maxpool3(x)

#         x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
#         x = self.rpe_conv(x)
#         x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
#         x = self.rpe_lif(x)
#         x = x + x_feat

#         x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
#         return x