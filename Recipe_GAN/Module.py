from Networks import Generator, Discriminator
from Data_Provider import Data_Provider
import torch
import torchvision.utils as vutils
from torch import nn, optim


class Module:

    def __init__(self, batch_size=64, noise_vector_size=100, num_epochs=1, lr=0.0002, beta1=0.5):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.data_provider = Data_Provider(batch_size)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.netG = Generator(noise_vector_size, self.data_provider.num_ingredients).to(self.device)
        self.netG.apply(self.weights_init)

        self.netD = Discriminator(self.data_provider.num_ingredients).to(self.device)
        self.netD.apply(self.weights_init)

        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(batch_size, noise_vector_size, device=self.device)
        self.noise_vector_size = noise_vector_size
        self.real_label = 1
        self.fake_label = 0

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))

        self.recipe_list = []

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train_module(self):
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.data_provider.dataloader, 0):
                self.netD.zero_grad()
                real_cpu = data.to(self.device).float()
                output = self.netD(real_cpu).view(-1)
                label = torch.full((self.batch_size,), self.real_label, device=self.device)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(self.batch_size, self.noise_vector_size, device=self.device)
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                output = self.netD(fake.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optimizerD.step()

                self.netG.zero_grad()
                label.fill_(self.real_label)
                output = self.netD(fake).view(-1)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.num_epochs, i, len(self.data_provider.dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if (iters % 500 == 0) or ((epoch == self.num_epochs - 1) and (i == len(self.data_provider.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    self.recipe_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
