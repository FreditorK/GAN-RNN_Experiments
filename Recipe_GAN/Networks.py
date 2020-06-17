from torch import nn
import numpy as np
import torch


class Generator(nn.Module):
    def __init__(self, noise_vector_size, num_ingredients):
        super(Generator, self).__init__()
        hid_lay_sizes = [
            noise_vector_size,
            np.floor_divide(num_ingredients, 4),
            np.floor_divide(num_ingredients, 2),
            num_ingredients
        ]

        assert noise_vector_size <= hid_lay_sizes[1]

        self.main = nn.Sequential(
            nn.Linear(hid_lay_sizes[0], hid_lay_sizes[1]),
            nn.ReLU(),
            nn.Linear(hid_lay_sizes[1], hid_lay_sizes[2]),
            nn.ReLU(),
            nn.Linear(hid_lay_sizes[2], hid_lay_sizes[3]),
            nn.Sigmoid()
        )

        try:
            self.load()
            print("Generator found")
        except:
            self.weights_init()
            print("No Generator was found")

    def forward(self, input):
        return torch.round(self.main(input))

    def weights_init(self):
        classname = self.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(self.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(self.weight.data, 1.0, 0.02)
            nn.init.constant_(self.bias.data, 0)

    def save(self):
        torch.save(self.state_dict(), 'generator.pth')

    def load(self):
        self.load_state_dict(torch.load('generator.pth'))


class Discriminator(nn.Module):
    def __init__(self, num_ingredients):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(num_ingredients, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        try:
            self.load()
            print("Discriminator found")
        except:
            self.weights_init()
            print("No Discriminator was found")

    def forward(self, input):
        return self.main(input)

    def weights_init(self):
        classname = self.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(self.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(self.weight.data, 1.0, 0.02)
            nn.init.constant_(self.bias.data, 0)

    def save(self):
        torch.save(self.state_dict(), 'discriminator.pth')

    def load(self):
        self.load_state_dict(torch.load('discriminator.pth'))
