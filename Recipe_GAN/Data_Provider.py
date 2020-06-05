import numpy as np
import pickle
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Data_Provider:

    def __init__(self):
        with np.load('recipes.npz', allow_pickle=True) as data:
            self.recipes = data['recipes']
            self.ingredients = data['ingredients']

        self.dataloader = torch.utils.data.DataLoader(RecipeDataset(self.recipes, len(self.ingredients)), batch_size=64, num_workers=2)
        print(next(iter(self.dataloader)))


class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, data, N):
        super(torch.utils.data.Dataset).__init__()
        self.data = data
        self.N = N

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        element = self.data[idx]
        vec = np.zeros(self.N)
        vec[element] = 1
        return vec


data_provider = Data_Provider()
