# convolution token embberding
# convolution projection
# attention
# MLP
# Complete model
import torch 
import torch.nn as nn
import torch.nn.functional as F

class CVCEmbedding(nn.Module):
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
    


def test_CVCTEmbedding():
    x = torch.randn(2, 3, 224, 224)  # Example input tensor
    model = CVCEmbedding(in_channels=3, embed_dim=64, patch_size=16, stride=16)
    output = model(x)
    print(output.shape)  

if __name__ == "__main__":
    test_CVCTEmbedding()
    # Expected output shape: (2, 196, 64) for a 224x224 input with patch size 16 and stride 16