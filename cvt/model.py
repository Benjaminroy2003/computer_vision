# convolution token embberding
# convolution projection
# attention
# MLP
# Complete model
import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from collections import OrderedDict
from einops import rearrange



class ConvEmbedding(nn.Module):
    def __init__(self, in_channels,embed_dim,patch_size,stride):
        super().__init__()
        self.proj= nn.Conv2d(in_channels,embed_dim,kernel_size = patch_size,stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,x):
        x = self.proj(x) # (B, C, H, W)
        x = x.flatten(2)
        x = x.transpose(1, 2) # (B, N, C)
        x = self.norm(x) 
        return x
    
class MultiheadAttention(nn.Module):

    def __init__(self, in_dim , out_dim ,num_heads, kernel_size=3, qkv_bias=False,stride_kv = 1, stride_q= 1, padding_kv = 1, padding_q= 1, with_cls_token=True):
        super().__init__()
        self.num_heads = num_heads
        self.padding = (kernel_size - 1) // 2
        self.scale = in_dim ** -0.5
        self.in_dim = in_dim
        self.head_dim = in_dim // num_heads
        self.kernel_size = kernel_size
        self.conv_proj_q = self.convolution_projection(in_dim,out_dim, kernel_size=kernel_size, padding=padding_q,stride=stride_q)
        self.conv_proj_k = self.convolution_projection(in_dim, out_dim,kernel_size=kernel_size, padding=padding_kv,stride=stride_kv)
        self.conv_proj_v = self.convolution_projection(in_dim, out_dim,kernel_size=kernel_size, padding=padding_kv,stride=stride_kv)

        self.proj_q = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.proj_k = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.proj_v = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.proj = nn.Linear(out_dim, out_dim)
    def convolution_projection(self, in_dim, out_dim, kernel_size, padding, stride ):
        proj = nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(
            in_dim,
            in_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
            groups=in_dim
        )),
        ('bn', nn.BatchNorm2d(in_dim)),
        ('rearrage', Rearrange('b c h w -> b (h w) c')),
        ]))
        return proj
    def conv_forward(self,x,h,w):
        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = x.flatten(2).transpose(1, 2)
        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = x.flatten(2).transpose(1, 2)
        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = x.flatten(2).transpose(1, 2)

        return q, k, v

    def forward(self,x,h,w):
        if self.conv_proj_q is not None or self.conv_proj_k is not None or self.conv_proj_v is not None:
            q,k,v = self.conv_forward(x,h,w)
        else:
            q= self.proj_q(x)
            k = self.proj_k(x)
            v = self.proj_v(x)
            q = q.flatten(2).transpose(1, 2)
            k = k.flatten(2).transpose(1, 2)
            v = v.flatten(2).transpose(1, 2)
        
        attention_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_score = F.softmax(attention_score, dim=-1)
        attention_output = torch.matmul(attention_score, v)
        attention_output = Rearrange(attention_output, 'b h w c -> b w (h c)')  

        x = self.proj(attention_output)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x 
           
class chat_gpt_multihead_attention(nn.Module):
            
    def __init__(self, in_dim, out_dim, num_heads, kernel_size=3, qkv_bias=False,
                 stride_kv=1, stride_q=1, padding_kv=1, padding_q=1, with_cls_token=True):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5  # scale should use head_dim, not in_dim

        # depthwise conv projections
        self.conv_proj_q = self.convolution_projection(in_dim, kernel_size, padding_q, stride_q)
        self.conv_proj_k = self.convolution_projection(in_dim, kernel_size, padding_kv, stride_kv)
        self.conv_proj_v = self.convolution_projection(in_dim, kernel_size, padding_kv, stride_kv)

        # linear projections
        self.proj_q = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.proj_k = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.proj_v = nn.Linear(in_dim, out_dim, bias=qkv_bias)

        self.proj_out = nn.Linear(out_dim, out_dim)

    def convolution_projection(self, in_dim, kernel_size, padding, stride):
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size,
                               padding=padding, stride=stride,
                               bias=False, groups=in_dim)),
            ('bn', nn.BatchNorm2d(in_dim)),
            ('rearrange', nn.Flatten(2))  # (B, C, H, W) -> (B, C, H*W)
        ]))

    def conv_forward(self, x):
        # x: (B, C, H, W)
        q = rearrange(self.conv_proj_q(x), 'b c n -> b n c')
        k = rearrange(self.conv_proj_k(x), 'b c n -> b n c')
        v = rearrange(self.conv_proj_v(x), 'b c n -> b n c')
        return q, k, v

    def forward(self, x):
        # x: (B, C, H, W)
        q, k, v = self.conv_forward(x)

        # linear proj
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        # split heads
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, H, N, D)

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj_out(out)
        return out

class Block(nn.Module):
    def __init__(self, in_dim ,out_dim, num_heads, kernel_size=3, qkv_bias=False,
                 stride_kv=1, stride_q=1, padding_kv=1, padding_q=1, with_cls_token=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        hidden_dim = out_dim * 4
        self.atten = MultiheadAttention( in_dim , out_dim ,num_heads, kernel_size, qkv_bias,stride_kv, stride_q, padding_kv, padding_q, with_cls_token)
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, hidden_dim, out_dim, dropout=0.)
    def forward(self,x,h,w):
        res = x
        x = self.norm(x)
        attn = self.atten(x,h,w)
        x = res + attn
        x = x + self.mlp(self.norm(x))
        return x
    
class VisisonTransformer(nn.Module):

    def __init__(self,
                 patch_size=16,
                 patch_stride = 16,
                 embed_dim=768,
                 in_chans=3,
                 depth=12,
                 num_heads=12,
                 qkv_bias = False,
                 init = 'trunc_norm',
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = ConvEmbedding(in_channels=in_chans,embed_dim=embed_dim,patch_size=patch_size,stride=patch_stride)
        block = []
        for i in range(depth):
            self.blocks = nn.ModuleList([Block(in_dim=embed_dim, out_dim=embed_dim, num_heads=num_heads, kernel_size=3, qkv_bias=qkv_bias)
                                         for _ in range(depth)])
        self.apply(self._init_weights)
    def forward(self,x):
        x = self.patch_embed(x)
        B,C,H,W = x.size()
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        for i ,blk in enumerate(self.blocks):
            x = blk(x,H,W)
        x = x.reshape(B, C, H, W)
        return x 
    def _init_weights(self, m):
        if isinstance(m,nn.Linear):
            nn.init.trunc_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,in_channel =3,num_classes=1000,init='trunc_norm',spec=None):
        super().__init__()
        self.num_classes = num_classes
        
def test_CVCTEmbedding():
    x = torch.randn(2, 3, 224, 224)  # Example input tensor
    model = ConvEmbedding(in_channels=3, embed_dim=64, patch_size=16, stride=16)
    output = model(x)
    print(output.shape)  

def test_attention():
    B, C, H, W = 2, 32, 8, 8   # batch, channels, height, width
    out_dim = 64
    num_heads = 8

    x = torch.randn(B, C, H, W)
    attn = MultiheadAttention(in_dim=C, out_dim=out_dim, num_heads=num_heads)

    out = attn(x)
    print("Input:", x.shape)   # (2, 32, 8, 8)
    print("Output:", out.shape)

if __name__ == "__main__":
    test_CVCTEmbedding()
    test_attention()
    # Expected output shape: (2, 196, 64) for a 224x224 input with patch size 16 and stride 16