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

    def forward(self, input):
        return torch.round(self.main(input))


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

    def forward(self, input):
        return self.main(input)
