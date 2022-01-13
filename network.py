import torch
import torch.nn as nn
from torch.nn import functional as F

class WA_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.rand(1, out_channel, in_channel, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)
        
        self.query = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel*in_channel*kernel_size*kernel_size, 1,1,0, bias=False)
            )
        #self.query[1].weight.data.fill_(1)
        nn.init.kaiming_normal_(self.query[1].weight)
        
        self.key = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel*in_channel*kernel_size*kernel_size, 1,1,0, bias=False)
            )
        #self.key[1].weight.data.fill_(1)
        nn.init.kaiming_normal_(self.key[1].weight)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch, in_channel, height, width = x.shape

        query = self.query(x) # batch, out_channel*in_channel*kernel_size*kernel_size, 1, 1
        key = self.key(x)
        #print(query.shape)
        attention = self.sigmoid(query+key)
        #print(attention.shape)

        attention = attention.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        #print(attention.shape)
        #print(self.weight.shape)
        weight = self.weight.repeat(batch,1,1,1,1) * attention
        #print(weight.shape)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        #print(weight.shape)
        
        x = x.view(1, batch * in_channel, height, width)
        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((2,3,32,32)).to(device)
    b = WA_Conv2d(3,16,3,2,1).to(device)
    c = b(a)
    print(c.shape)
    
    