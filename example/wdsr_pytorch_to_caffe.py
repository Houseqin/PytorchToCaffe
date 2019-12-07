import sys

sys.path.insert(0, '.')
import os
import numpy as np
import torch
import pytorch_to_caffe
from example.wdsr_b_nown import WDSR_B_NOWN
from example.rdn import RDN
from torchvision import transforms

if __name__ == '__main__':
    use_gpu = False
    ckpt_path = './checkpoints/8_32_nomp_whiten10_tests/'
    ckpt_path = os.path.join(ckpt_path, 'net.pt')
    name = 'WDSR'
    net = WDSR_B_NOWN(n_blocks=8, n_feats=32, n_colors=1, scale=1, rgb_mean=[0.5])
    # net = RDN(channel = 1, growth_rate = 16, rdb_number = 2)
    net = torch.nn.DataParallel(net, use_gpu)

    if use_gpu:
        device = torch.device('cuda:0,1')
        input = input.to(device)
        net = net.to(device)
        checkpoint = torch.load(ckpt_path)['net']
    else:
        checkpoint = torch.load("/Users/momo/Desktop/net.pt", 'cpu')['net']
        # checkpoint = torch.load("/Users/momo/Desktop/rdn_net.pt", 'cpu')['net']

    net.load_state_dict(checkpoint)
    net.eval()

    if use_gpu:
        img = np.load('/gpu001/qinsihao/data/38/test/org/dengziqi1_2201.npy')[:1080 * 1920].reshape([1080, 1920])
    else:
        img = np.load('/Users/momo/Desktop/dengziqi2_1200_org.npy')[:1080 * 1920].reshape([1080, 1920])

    transform = transforms.Compose([transforms.ToTensor()])
    input = transform(img)
    input_un = input.unsqueeze(0)
    pytorch_to_caffe.trans_net(net, input.unsqueeze(0), name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))