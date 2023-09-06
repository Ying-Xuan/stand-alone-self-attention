import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Union

class Attention(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, kernel_size:int=3, stride:Union[int, tuple]=2,  groups:int=1, bias=False):
        super(Attention, self).__init__()
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        if type(stride) is int:
            self.stride=(stride, stride)
        else:
            self.stride=stride
        self.padding = kernel_size//2
        self.groups = groups

        assert self.out_channel % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        # relative position
        self.rel = nn.Parameter(torch.randn(out_channel, 1, 1, kernel_size, kernel_size), requires_grad=True)
        init.normal_(self.rel, 0, 1)

        # point-wise convolution (1x1 conv)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=bias)

    def forward(self, x):

        batch, channels, height, width = x.size()
        
        # padding for dim=-1,-2 (i.e. height width)
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q = self.conv(x)
        k = self.conv(padded_x)
        v = self.conv(padded_x)

        # (B,C,H+2p,W+2p) => (B,C,Floor((H+2p-k)/s)+1,Floor((W+2p-k)/s)+1, k,k)
        q = q.unfold(2, 1, self.stride[0]).unfold(3, 1, self.stride[1])
        k = k.unfold(2, self.kernel_size, self.stride[0]).unfold(3, self.kernel_size, self.stride[1])
        v= v.unfold(2, self.kernel_size, self.stride[0]).unfold(3, self.kernel_size, self.stride[1])

        batch, channels, height, width,_,_ = k.size()

        # add relative position
        k = k + self.rel

        # split group
        # (B,C,Floor((H+2p-k)/s)+1,Floor((W+2p-k)/s)+1,k,k) => (B, g , C/g, Floor((H+2p-k)/s)+1,Floor((W+2p-k)/s)+1,k,k)
        k = k.contiguous().view(batch, self.groups, self.out_channel//self.groups, height, width, self.kernel_size,self.kernel_size)
        v = v.contiguous().view(batch, self.groups, self.out_channel//self.groups, height, width, self.kernel_size,self.kernel_size)
        # (B,C,H,W) => # (B, g, C/g,H,W, 1) 
        q = q.view(batch, self.groups, self.out_channel//self.groups, height, width, 1, -1)
        
        # point-wise multiplication on dim=-1
        out = q * k
        out = F.softmax(out, dim=-1)
        # point-wise multiplication and sum on dim=-1
        # concatenate all head
        out = torch.einsum('bnchwkk,bnchwkk -> bnchw', out, v).view(batch, -1, height, width)

        return out
    

def main():

    x= torch.randn(1,3,18,256)
    att = Attention(in_channel=3, out_channel=5, kernel_size=3, stride=(1,2))
    y = att(x)
    print(y.shape)  # (1,5,224,224)


if __name__=='__main__':
    main()