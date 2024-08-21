import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import mlflow
import mlflow.pytorch

# Define the progressive Self Atention Generative Adversarial Network for generator and discriminator (SAGAN)

# Data params
data_mean = 4
data_stddev = 1.25

# Model params
g_input_size = 100     # Random noise dimension coming into generator
g_hidden_size = 128   # Generator complexity
g_output_size = 1    # Size of generated output vector
d_input_size = 100   # Minibatch size
d_hidden_size = 128   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake' classification
minibatch_size = d_input_size

# Function to generate real data
def get_real_data():
    return torch.Tensor(np.random.normal(data_mean, data_stddev, (50, minibatch_size)))

# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out, attention

# Generator model with Self-Attention
class SAGANGenerator(nn.Module):
    def __init__(self):
        super(SAGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(g_input_size, g_hidden_size * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_hidden_size * 4),
            nn.ReLU(True),
            SelfAttention(g_hidden_size * 4),
            nn.ConvTranspose2d(g_hidden_size * 4, g_hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_hidden_size * 2, g_hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_hidden_size, g_output_size, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator model with Self-Attention
class SAGANDiscriminator(nn.Module):
    def __init__(self):
        super(SAGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(g_output_size, g_hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(g_hidden_size, g_hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(g_hidden_size * 2),
            nn.Conv2d(g_hidden_size * 2, g_hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(g_hidden_size * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)

# Training loop with MLflow logging
def train():
    # Initialize models and optimizers
    G = SAGANGenerator()
    D = SAGANDiscriminator()
    criterion = nn.BCELoss()
    d_optimizer = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Start MLflow experiment
    mlflow.set_experiment("SAGAN Training")
    with mlflow.start_run():
        for epoch in range(1000):
            # 1. Train Discriminator
            real_data = get_real_data().view(-1, g_output_size, 1, 1)  # Reshape for Conv layers
            fake_data = G(Variable(torch.randn(minibatch_size, g_input_size, 1, 1)))
            d_real_decision = D(real_data)
            d_fake_decision = D(fake_data.detach())

            d_real_error = criterion(d_real_decision, Variable(torch.ones(minibatch_size, 1)))  
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(minibatch_size, 1))) 
            d_optimizer.zero_grad()
            d_error = d_real_error + d_fake_error
            d_error.backward()
            d_optimizer.step()

            # 2. Train Generator
            fake_data = G(Variable(torch.randn(minibatch_size, g_input_size, 1, 1)))
            d_fake_decision = D(fake_data)
            g_error = criterion(d_fake_decision, Variable(torch.ones(minibatch_size, 1))) 
            g_optimizer.zero_grad()
            g_error.backward()
            g_optimizer.step()

            # Log results and print every 100 epochs
            if epoch % 100 == 0:
                print("Epoch %s: D (%s real_err, %s fake_err) G (%s err)" % (epoch, d_real_error.item(), d_fake_error.item(), g_error.item()))

                # Log errors to MLflow
                mlflow.log_metric("d_real_error", d_real_error.item(), step=epoch)
                mlflow.log_metric("d_fake_error", d_fake_error.item(), step=epoch)
                mlflow.log_metric("g_error", g_error.item(), step=epoch)

        # Log model parameters and final weights
        mlflow.log_param("g_input_size", g_input_size)
        mlflow.log_param("g_hidden_size", g_hidden_size)
        mlflow.log_param("g_output_size", g_output_size)
        mlflow.log_param("d_input_size", d_input_size)
        mlflow.log_param("d_hidden_size", d_hidden_size)
        mlflow.log_param("d_output_size", d_output_size)

        # Log the final models
        mlflow.pytorch.log_model(G, "generator")
        mlflow.pytorch.log_model(D, "discriminator")

train()
