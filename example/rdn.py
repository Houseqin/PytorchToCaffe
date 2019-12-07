import torch
import torch.nn as nn
import torch.nn.init as init
import math


class RDN(nn.Module):
    def __init__(self, channel, growth_rate, rdb_number):
        super(RDN, self).__init__()
        self.SFF1 = nn.Conv2d(in_channels=channel, out_channels=16, kernel_size=3, padding=1, stride=2)
        self.SFF2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=2)
        self.RDB1 = RDB(nb_layers=rdb_number, input_dim=16, growth_rate=16)
        self.RDB2 = RDB(nb_layers=rdb_number, input_dim=16, growth_rate=16)
        self.RDB3 = RDB(nb_layers=rdb_number, input_dim=16, growth_rate=16)
        self.GFF1 = nn.Conv2d(in_channels=16 * 3, out_channels=16, kernel_size=1, padding=0)
        self.GFF2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        # self.upconv = nn.Conv2d(in_channels = 8, out_channels=(8*upscale_factor*upscale_factor),kernel_size = 3,padding = 1)
        # self.pixelshuffle = nn.PixelShuffle(upscale_factor)
        self.upconv = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2,
                                         padding=0)  # ,output_padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=channel, kernel_size=2, stride=2, padding=0)


    def forward(self, x):
        f_ = self.SFF1(x)
        f_0 = self.SFF2(f_)
        f_1 = self.RDB1(f_0)
        f_2 = self.RDB2(f_1)
        f_3 = self.RDB3(f_2)
        f_D = torch.cat((f_1, f_2, f_3), 1)
        f_1x1 = self.GFF1(f_D)
        f_GF = self.GFF2(f_1x1)
        f_DF = f_GF + f_0
        f_upconv = self.upconv(f_DF)
        # f_upscale = self.pixelshuffle(f_upconv)
        # f_conv2 = self.conv2(f_upscale)
        f_conv2 = self.conv2(f_upconv)
        return f_conv2 + x


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x, out), 1)


class BasicBlock1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock1, self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, padding=0, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, nb_layers, input_dim, growth_rate):
        super(RDB, self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers, input_dim, growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels=input_dim + nb_layers * growth_rate, \
                                 out_channels=growth_rate, \
                                 kernel_size=1, \
                                 stride=1, \
                                 padding=0)

    def _make_layer(self, nb_layers, input_dim, growth_rate):
        layers = []
        for i in range(nb_layers):
            #	    if nb_layers-i>2:
            layers.append(BasicBlock(input_dim + i * growth_rate, growth_rate))
        #           else:
        #		layers.append(BasicBlock1(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        out = self.conv1x1(out)
        return out + x

if __name__ == '__main__':
    import time

    net = RDN(channel=1, growth_rate=16, rdb_number=2)

    input = torch.randn((1, 1, 360, 640))
    start = time.time()
    output = net(input)
    print(f'Total forward time: {time.time() - start:.4f}s')

    summary(net, x=input)

    print(net)
    print("output' Size: {}".format(output.size()))

# dummy_input = torch.rand(256, 1, 64, 64) #假设输入13张1*28*28的图片
# model = RDN(channel = 1, growth_rate = 16, rdb_number = 2)
# with SummaryWriter(comment='RDN') as w:
#     w.add_graph(model, (dummy_input, ))