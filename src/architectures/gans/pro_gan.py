import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import mlflow
import mlflow.pytorch


# Define the progressive growing architecture for generator and discriminator (PRO GAN)

latent_dim = 128
start_resolution = 4
max_resolution = 64
epochs_per_stage = 100

class Generator(nn.Module):
    def __init__(self, latent_dim, start_resolution, max_resolution):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.start_resolution = start_resolution
        self.max_resolution = max_resolution
        self.progressive_layers = nn.ModuleList()

        resolution = start_resolution
        while resolution <= max_resolution:
            self.progressive_layers.append(self._make_layer(resolution))
            resolution *= 2

    def _make_layer(self, resolution):
        return nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, resolution, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(resolution),
            nn.ReLU()
        )

    def forward(self, x, stage):
        for i in range(stage + 1):
            x = self.progressive_layers[i](x)
        return x


class Discriminator(nn.Module):
    def __init__(self, start_resolution, max_resolution):
        super(Discriminator, self).__init__()
        self.start_resolution = start_resolution
        self.max_resolution = max_resolution
        self.progressive_layers = nn.ModuleList()

        resolution = start_resolution
        while resolution <= max_resolution:
            self.progressive_layers.append(self._make_layer(resolution))
            resolution *= 2

    def _make_layer(self, resolution):
        return nn.Sequential(
            nn.Conv2d(resolution, resolution * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(resolution * 2),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, stage):
        for i in range(stage + 1):
            x = self.progressive_layers[i](x)
        return x

# Function to generate real data
def get_real_data(resolution, batch_size):
    return torch.Tensor(np.random.normal(0, 1, (batch_size, resolution, resolution)))

# Training loop with progressive growing
def train_progan(latent_dim, start_resolution, max_resolution, epochs_per_stage):
    G = Generator(latent_dim, start_resolution, max_resolution)
    D = Discriminator(start_resolution, max_resolution)
    criterion = nn.BCELoss()
    g_optimizer = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    mlflow.set_experiment("ProGAN Training")
    with mlflow.start_run():
        stage = 0
        current_resolution = start_resolution

        while current_resolution <= max_resolution:
            for epoch in range(epochs_per_stage):
                real_data = get_real_data(current_resolution, 64)
                fake_data = G(torch.randn(64, latent_dim, 1, 1), stage)

                d_real_decision = D(real_data, stage)
                d_fake_decision = D(fake_data.detach(), stage)

                d_real_error = criterion(d_real_decision, torch.ones_like(d_real_decision))
                d_fake_error = criterion(d_fake_decision, torch.zeros_like(d_fake_decision))

                d_optimizer.zero_grad()
                d_error = d_real_error + d_fake_error
                d_error.backward()
                d_optimizer.step()

                fake_data = G(torch.randn(64, latent_dim, 1, 1), stage)
                d_fake_decision = D(fake_data, stage)
                g_error = criterion(d_fake_decision, torch.ones_like(d_fake_decision))

                g_optimizer.zero_grad()
                g_error.backward()
                g_optimizer.step()

                if epoch % 10 == 0:
                    print(f"Stage {stage} Epoch {epoch}: D Error: {d_error.item()} G Error: {g_error.item()}")
                    mlflow.log_metric(f"d_error_stage_{stage}", d_error.item(), step=epoch)
                    mlflow.log_metric(f"g_error_stage_{stage}", g_error.item(), step=epoch)

            mlflow.log_param(f"resolution_stage_{stage}", current_resolution)
            current_resolution *= 2
            stage += 1

        # Log final models
        mlflow.pytorch.log_model(G, "generator")
        mlflow.pytorch.log_model(D, "discriminator")

train_progan(latent_dim, start_resolution, max_resolution, epochs_per_stage)
