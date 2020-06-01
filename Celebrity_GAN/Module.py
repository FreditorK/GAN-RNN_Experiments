from Networks import Generator, Discriminator
import torch
from torch import nn


class Module:

    def __init__(self):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        netG = Generator().to(device)
        netG.apply(self.weights_init)

        netD = Discriminator(device).to(device)
        netD.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)