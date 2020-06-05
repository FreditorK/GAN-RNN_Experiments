import torch.nn.parallel
import torch.utils.data
import numpy as np


class Data_Provider:

    def __init__(self, batch_size):
        with np.load('recipes.npz', allow_pickle=True) as data:
            recipes = data['recipes']
            ingredients = data['ingredients']

        self.num_ingredients = len(ingredients)
        self.dataloader = torch.utils.data.DataLoader(Recipe_Dataset(recipes, self.num_ingredients), batch_size=batch_size, num_workers=2)


class Recipe_Dataset(torch.utils.data.Dataset):
    def __init__(self, recipes, num_ingredients):
        super(torch.utils.data.Dataset).__init__()
        self.data = recipes
        self.num_ingredients = num_ingredients

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        element = self.data[idx].astype(int)
        vec = np.zeros(self.num_ingredients)
        vec[element] = 1
        return vec

