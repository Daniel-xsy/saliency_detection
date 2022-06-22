import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(VGGBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_chans)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class Maxpool(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(Maxpool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)(x)


class MLNet(nn.Module):
    def __init__(self, img_size=256):
        super(MLNet, self).__init__()

        self.block_1 = nn.Sequential(VGGBlock(3, 64), Maxpool(2, 2))
        self.block_2 = nn.Sequential(VGGBlock(64, 128), Maxpool(2, 2))
        self.block_3 = nn.Sequential(VGGBlock(128, 256), Maxpool(2, 2))
        self.block_4 = nn.Sequential(VGGBlock(256, 512), Maxpool(3, 1, 1))
        self.block_5 = nn.Sequential(VGGBlock(512, 512))

        self.conv_head = nn.Sequential(
            nn.Conv2d(256+512+512, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, 1)
        )

        self.dp = nn.Dropout()

        self.prior = nn.Parameter(torch.randn(img_size, img_size), requires_grad=True)

    def _upsampling(self, x, out_size):

        B = x.size(0)
        device = x.device

        if isinstance(out_size, int):
            out_h = out_w = out_size
        elif isinstance(out_size, tuple):
            out_h, out_w = out_size

        new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
        new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
        grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1).to(device)

        output = F.grid_sample(x, grid=grid, mode='bilinear')

        return output


    def forward(self, x): 

        B, C, H, W = x.size()

        x = self.block_1(x)
        x = self.block_2(x)
        x_3 = self.block_3(x)
        x_4 = self.block_4(x_3)
        x_5 = self.block_5(x_4)

        # cat multi scale feature
        x = torch.cat((x_3, x_4, x_5), dim=1)
        x = self.dp(x)
        x = self.conv_head(x)

        # recover original size
        x = self._upsampling(x, (H,W))
        # multiply prior matrix
        x = x * self.prior[None]

        return x