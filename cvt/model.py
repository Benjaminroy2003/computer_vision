import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange

class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W)
        x = x.flatten(2)  # (B, C, H*W)
        x = x.transpose(1, 2)  # (B, N, C)
        x = self.norm(x) 
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, kernel_size=3, qkv_bias=False,
                 stride_kv=1, stride_q=1, padding_kv=1, padding_q=1, with_cls_token=True):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        
        # Convolutional projections
        self.conv_proj_q = self.convolution_projection(in_dim, kernel_size, padding_q, stride_q)
        self.conv_proj_k = self.convolution_projection(in_dim, kernel_size, padding_kv, stride_kv)
        self.conv_proj_v = self.convolution_projection(in_dim, kernel_size, padding_kv, stride_kv)

        # Linear projections
        self.proj_q = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.proj_k = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.proj_v = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.proj = nn.Linear(out_dim, out_dim)

    def convolution_projection(self, in_dim, kernel_size, padding, stride):
        proj = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                in_dim, in_dim,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
                groups=in_dim
            )),
            ('bn', nn.BatchNorm2d(in_dim)),
            ('rearrange', nn.Flatten(2))  # (B, C, H, W) -> (B, C, H*W)
        ]))
        return proj

    def conv_forward(self, x, h, w):
        # Reshape back to spatial format for conv operations
        x_spatial = x.transpose(1, 2).reshape(-1, self.in_dim, h, w)
        
        q = rearrange(self.conv_proj_q(x_spatial), 'b c n -> b n c')
        k = rearrange(self.conv_proj_k(x_spatial), 'b c n -> b n c')
        v = rearrange(self.conv_proj_v(x_spatial), 'b c n -> b n c')
        
        return q, k, v

    def forward(self, x, h, w):
        # x: (B, N, C) where N = H*W
        q, k, v = self.conv_forward(x, h, w)
        
        # Apply linear projections
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # Compute attention
        attention_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_score = F.softmax(attention_score, dim=-1)
        attention_output = torch.matmul(attention_score, v)
        
        # Merge heads
        attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')
        
        # Final projection
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

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, kernel_size=3, qkv_bias=False,
                 stride_kv=1, stride_q=1, padding_kv=1, padding_q=1, with_cls_token=False,
                 mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        hidden_dim = int(out_dim * mlp_ratio)
        
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = MultiheadAttention(in_dim, out_dim, num_heads, kernel_size, qkv_bias,
                                     stride_kv, stride_q, padding_kv, padding_q, with_cls_token)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = MLP(out_dim, hidden_dim, out_dim, dropout=dropout)
        
        # Projection layer if dimensions don't match
        self.proj = None
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x, h, w):
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, h, w)
        
        # Residual connection with projection if needed
        if self.proj is not None:
            x = self.proj(x)
        x = x + attn_out
        
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 embed_dim=768,
                 in_chans=3,
                 depth=12,
                 num_heads=12,
                 qkv_bias=False,
                 mlp_ratio=4.0,
                 dropout=0.,
                 init='trunc_norm'):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = ConvEmbedding(in_channels=in_chans, embed_dim=embed_dim,
                                       patch_size=patch_size, stride=patch_stride)
        
        self.blocks = nn.ModuleList([
            Block(in_dim=embed_dim, out_dim=embed_dim, num_heads=num_heads,
                  kernel_size=3, qkv_bias=qkv_bias, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, C)
        
        # Calculate spatial dimensions after patch embedding
        B, N, C = x.shape
        # Assuming square images, calculate H and W
        H = W = int(N ** 0.5)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x, H, W)
        
        x = self.norm(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, init='trunc_norm', spec=None):
        super().__init__()
        self.num_classes = num_classes
        
        # Default specification if none provided
        if spec is None:
            spec = {
                'INIT': 'trunc_norm',
                'NUM_STAGES': 3,
                'PATCH_SIZE': [7, 3, 3],
                'PATCH_STRIDE': [4, 2, 2],
                'PATCH_PADDING': [2, 1, 1],
                'DIM_EMBED': [64, 192, 384],
                'NUM_HEADS': [1, 3, 6],
                'DEPTH': [1, 2, 10],
                'MLP_RATIO': [4.0, 4.0, 4.0],
                'QKV_BIAS': [True, True, True],
                'DROPOUT_RATE': [0.0, 0.0, 0.0],
                'DROP_PATH_RATE': [0.0, 0.0, 0.1],
                'CLS_TOKEN': [False, False, True]
            }
        
        self.spec = spec
        self.num_stages = spec['NUM_STAGES']
        
        # Build stages
        self.stages = nn.ModuleList()
        
        for i in range(self.num_stages):
            if i == 0:
                stage_in_chans = in_channels
            else:
                stage_in_chans = spec['DIM_EMBED'][i-1]
            
            stage = VisionTransformer(
                patch_size=spec['PATCH_SIZE'][i],
                patch_stride=spec['PATCH_STRIDE'][i],
                embed_dim=spec['DIM_EMBED'][i],
                in_chans=stage_in_chans,
                depth=spec['DEPTH'][i],
                num_heads=spec['NUM_HEADS'][i],
                qkv_bias=spec['QKV_BIAS'][i],
                mlp_ratio=spec['MLP_RATIO'][i],
                dropout=spec['DROPOUT_RATE'][i],
                init=init
            )
            self.stages.append(stage)
        
        # Classification head
        self.head = nn.Linear(spec['DIM_EMBED'][-1], num_classes)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.apply(self._init_weights)

    def forward(self, x):
        B = x.shape[0]
        
        for i, stage in enumerate(self.stages):
            x = stage(x)  # (B, N, C)
            
            # Prepare input for next stage (reshape to spatial format)
            if i < self.num_stages - 1:
                N, C = x.shape[1], x.shape[2]
                H = W = int(N ** 0.5)
                x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Global average pooling and classification
        x = x.transpose(1, 2)  # (B, C, N)
        x = self.avgpool(x).squeeze(-1)  # (B, C)
        x = self.head(x)  # (B, num_classes)
        
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

# Example usage
def create_cvt_small(num_classes=1000):
    """Create a small CvT model"""
    spec = {
        'INIT': 'trunc_norm',
        'NUM_STAGES': 3,
        'PATCH_SIZE': [7, 3, 3],
        'PATCH_STRIDE': [4, 2, 2],
        'PATCH_PADDING': [2, 1, 1],
        'DIM_EMBED': [64, 192, 384],
        'NUM_HEADS': [1, 3, 6],
        'DEPTH': [1, 2, 10],
        'MLP_RATIO': [4.0, 4.0, 4.0],
        'QKV_BIAS': [True, True, True],
        'DROPOUT_RATE': [0.0, 0.0, 0.0],
        'DROP_PATH_RATE': [0.0, 0.0, 0.1],
        'CLS_TOKEN': [False, False, True]
    }
    return ConvolutionalVisionTransformer(num_classes=num_classes, spec=spec)

def create_cvt_base(num_classes=1000):
    """Create a base CvT model"""
    spec = {
        'INIT': 'trunc_norm',
        'NUM_STAGES': 3,
        'PATCH_SIZE': [7, 3, 3],
        'PATCH_STRIDE': [4, 2, 2],
        'PATCH_PADDING': [2, 1, 1],
        'DIM_EMBED': [96, 288, 576],
        'NUM_HEADS': [1, 3, 6],
        'DEPTH': [2, 4, 16],
        'MLP_RATIO': [4.0, 4.0, 4.0],
        'QKV_BIAS': [True, True, True],
        'DROPOUT_RATE': [0.0, 0.0, 0.0],
        'DROP_PATH_RATE': [0.0, 0.0, 0.1],
        'CLS_TOKEN': [False, False, True]
    }
    return ConvolutionalVisionTransformer(num_classes=num_classes, spec=spec)

# Test the model
if __name__ == "__main__":
    # Create model
    model = create_cvt_small(num_classes=1000)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")