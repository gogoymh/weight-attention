import torch
import torch.nn as nn
from torch.nn import functional as F

class ECA_Layer(nn.Module):
    def __init__(self, channels, kernel=3):
        super(ECA_Layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel, padding=(kernel-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        
        return x*y.expand_as(x)

class Attention(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3):
        super().__init__()
        
        self.inter_channel = in_channel // 2
        
        self.query = nn.Conv2d(in_channel, self.inter_channel, 1, 4, 0, bias=False)
        self.key = nn.Conv2d(in_channel, self.inter_channel, 1, 4, 0, bias=False)
        self.value = nn.Conv2d(in_channel, self.inter_channel, 1, 4, 0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel, padding=(kernel-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.conv2d = nn.Conv2d(self.inter_channel, out_channel, 3, 1, 1, bias=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        query = self.query(x).reshape(B, self.inter_channel, -1).permute(0,2,1)
        print(query.shape)
        key = self.key(x).reshape(B, self.inter_channel, -1)
        print(key.shape)
        gate = self.softmax(torch.matmul(query,key))
        print(gate.shape)
        value = self.value(x).reshape(B, self.inter_channel, -1).permute(0,2,1)
        print(value.shape)
        gated_value = torch.matmul(gate, value).permute(0,2,1).reshape(B, self.inter_channel, H//4, W//4)
        print(gated_value.shape)
        
        channel_attention = self.avg_pool(gated_value)
        channel_attention = self.conv1d(channel_attention.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        channel_attention = self.sigmoid(channel_attention)
        
        out = gated_value*channel_attention.expand_as(gated_value)
        out = self.conv2d(out)
        
        return out, gate, channel_attention
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((1,8,32,32)).to(device)
    b = Attention(8,8).to(device)
    c, d, e = b(a)
    print(c.shape)
    
    d = d.squeeze().detach().cpu().numpy()
    plt.imshow(d, cmap="gray")
    plt.show()
    plt.close()
    
    
    
    