import torch
import torch.nn as nn
from common import MLP, weighted_mse_loss, weighted_sumrate_loss, SumRateLoss
import numpy as np
import matplotlib.pyplot as plt
from Paras import args


def conv1d_5(inplanes, outplanes, stride=1, bias=False):
    return nn.Conv1d(inplanes, outplanes, kernel_size=5, stride=stride,
                     padding=2, bias=bias)

class Block(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, downsample=None, bn=False):
        super(Block, self).__init__()
        self.bn = bn

        self.conv1 = conv1d_5(inplanes, outplanes, stride)
        self.bn1 = nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = conv1d_5(outplanes, outplanes)
        self.bn2 = nn.BatchNorm1d(outplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=5, stride=5, padding=0):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels=inplanes, out_channels=outplanes,
                                           padding=padding, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv1 = conv1d_5(inplanes, outplanes)
        self.relu = nn.ReLU()
        self.conv2 = conv1d_5(outplanes, outplanes)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        out = torch.cat((x1, x2), dim=1)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, inplanes=1, layers=(2, 2, 2, 2)):
        super(ResNet, self).__init__()
        self.inplanes = inplanes

        self.encoder1 = self._make_encoder(Block, 32, layers[0], 5)
        self.encoder2 = self._make_encoder(Block, 64, layers[1], 5)
        self.encoder3 = self._make_encoder(Block, 128, layers[2], 5)
        self.encoder4 = self._make_encoder(Block, 256, layers[3], 4)

        self.decoder3 = DecoderBlock(256, 128, stride=4, kernel_size=4, padding=1)
        self.decoder2 = DecoderBlock(128, 64, padding=1)
        self.decoder1 = DecoderBlock(64, 32)

        self.conv1x1 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=5, bias=False)

    def _make_encoder(self, block, planes, blocks, stride=1):
        downsample = None
        if self.inplanes != planes or stride != 1:
            downsample = nn.Conv1d(self.inplanes, planes, stride=stride, kernel_size=1, bias=False)
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        down1 = self.encoder1(x)
        down2 = self.encoder2(down1)
        down3 = self.encoder3(down2)
        down4 = self.encoder4(down3)

        up3 = self.decoder3(down4, down3)
        up2 = self.decoder2(up3, down2)
        up1 = self.decoder1(up2, down1)
        out = self.conv1x1(up1)

        return out

def cdim(inputsize, chanel=1):
    ts = torch.randn(inputsize)
    net = ResNet(inplanes=chanel)
    out = net(ts)
    return np.prod(out.shape)


def get_batch(x, y):
    ll = len(x)
    set_x, set_y = torch.reshape(x, (ll, 1, -1)), y

    return set_x, set_y



class Net(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args,
                 ):
        super(Net, self).__init__()

        self.n_outputs = n_outputs
        self.net = ResNet(inplanes=1)
        dim = cdim((1, 1, args.user * args.user), 1)
        self.fnet = MLP([dim] + [64, 32] + [n_outputs])


        # setup optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=args.lr)
        self.n_iter = args.n_iter
        self.mini_batch_size = args.mini_batch_size

        # setup losses
        self.noise = args.noise
        self.loss_wmse = weighted_mse_loss
        self.loss_wsumrate = weighted_sumrate_loss

        self.x = 0
        self.ax = []
        self.ay = []


    def forward(self, x, t):
        x, _ = get_batch(x, None)
        o_put = self.net(x)
        o_put = torch.squeeze(o_put, 1)
        o_put = self.fnet(o_put)
        return o_put.reshape(-1, self.n_outputs)


    def observe(self, x, t, y, loss_type='MSE', x_te=None, x_tr=None):
        self.train()
        if loss_type == 'MSE':
            loss = torch.nn.MSELoss()
        else:
            loss = SumRateLoss
        set_x, set_y = get_batch(x, y)

        # plt.ion()
        for epoch in range(self.n_iter):
            permutation = torch.randperm(set_x.size(0))
            for i in range(0, set_x.size(0), self.mini_batch_size):
                self.zero_grad()
                indices = permutation[i: i+self.mini_batch_size]
                mini_batch_x = set_x[indices]
                mini_batch_y = set_y[indices]

                out = self.forward(mini_batch_x, t)

                if loss_type == 'MSE':
                    ptloss = loss(out, mini_batch_y)
                else:
                    ptloss = loss(mini_batch_x, out, args.noise)
                ptloss.backward()
                self.opt.step()

                self.x = self.x + 1
                if 1:  # ptloss.detach().numpy() < 0.18:
                    plt.clf()
                    self.ax.append(self.x)
                    self.ay.append(ptloss.cpu().detach().numpy())

            plt.plot(self.ax, self.ay, '-', color='r', linewidth=0.5)
            plt.draw()
            plt.savefig('results/train_loss.png')









                



