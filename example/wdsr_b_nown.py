# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class WDSR_B_NOWN(nn.Module):
    def __init__(self, scale, n_blocks, n_feats, rgb_mean,
                 cuda=False, n_colors=3, res_scale=1.0,
                 kernel_size=3, act=nn.ReLU(True)):
        super(WDSR_B_NOWN, self).__init__()
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            rgb_mean)).view([1, n_colors, 1, 1])
        if cuda:
            self.register_buffer('rgbmean', self.rgb_mean.cuda())
        else:
            self.rgbmean = self.rgb_mean
            # self.register_buffer('rgbmean', self.rgb_mean)

        # define head module
        head = []
        head.append(
            nn.Conv2d(n_colors, n_feats, 3, stride=2, padding=3//2))
        # DOWN
        head.append(
            nn.Conv2d(n_feats, n_feats, 3, stride=2, padding=3 // 2))

        # define body module
        body = []
        for i in range(n_blocks):
            body.append(
                Block(n_feats, kernel_size, act=act, res_scale=res_scale))

        # define tail module
        tail = []
        out_feats = scale*scale*n_colors

        # UP
        tail.append(
            nn.ConvTranspose2d(n_feats, n_feats, kernel_size=2, stride=2)
        )
        tail.append(
            nn.Conv2d(n_feats, out_feats, 3, padding=3//2)
        )
        tail.append(
            nn.ConvTranspose2d(out_feats, out_feats, kernel_size=2, stride=2)
        )

        skip = []
        skip.append(
            nn.Conv2d(n_colors, out_feats, 5, padding=5//2)
        )

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = (x - self.rgbmean)*2.
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        # x = torch.clamp(x, -1.0, 1.0)
        x = x*0.5 + self.rgbmean
        x = torch.clamp(x, 0.0, 1.0)
        return x
class Block(nn.Module):
    def __init__(self, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1.0):
        super(Block, self).__init__()
        self.res_scale = res_scale

        body = []
        expand = 6
        linear = 0.8
        body.append(
            nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2))
        body.append(act)
        body.append(
            nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2))
        body.append(
            nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

if __name__ == '__main__':
    from torchsummaryX import summary
    import time

    wdsr_params = {
        'n_block': 8,
        'n_feat': 16,
        'n_color': 1,
        'scale': 1,
        'rgb_mean': [0.5]
    }

    # net = WDSR_B_NOWN(n_blocks=8, n_feats=16, n_colors=1, scale=1, rgb_mean=[0.5])
    net = WDSR_B_NOWN(n_blocks=wdsr_params['n_block'], n_feats=wdsr_params['n_feat'], n_colors=wdsr_params['n_color'], scale=wdsr_params['scale'], rgb_mean=wdsr_params['rgb_mean'])


    input = torch.randn((1, 1, 360, 640))
    start = time.time()
    output = net(input)
    print(f'Total forward time: {time.time() - start:.4f}s')

    summary(net, x=input)

    print(net)
    print("output' Size: {}".format(output.size()))