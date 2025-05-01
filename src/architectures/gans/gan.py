import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import mlflow
import mlflow.pytorch

# Data params
data_mean = 4
data_stddev = 1.25

# Model params
g_input_size = 50     # Random noise dimension coming into generator
g_hidden_size = 100   # Generator complexity
g_output_size = 50   # Size of generated output vector
d_input_size = 50   # Minibatch size
d_hidden_size = 50   # Discriminator complexity
d_hidden_2_size = 50 # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake' classification
minibatch_size = d_input_size

# Function to generate real data
def get_real_data():
    return torch.Tensor(np.random.normal(data_mean, data_stddev, (50, minibatch_size)))

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.LSTM(g_input_size, g_hidden_size),
            nn.ReLU(),
            nn.Linear(g_hidden_size, g_output_size)
        )
    
    def forward(self, x):
        return self.model(x)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_input_size, d_hidden_size),
            nn.ReLU(),
            nn.Linear(d_hidden_size, d_hidden_2_size),
            nn.ReLU(),
            nn.Linear(d_hidden_2_size, d_output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Training loop with MLflow logging
def train():
    # Initialize models and optimizers
    gen_nn = Generator()
    disc_nn = Discriminator()
    criterion = nn.BCELoss()
    d_optimizer = Adam(gen_nn.parameters())
    g_optimizer = Adam(disc_nn.parameters())

    # Start MLflow experiment
    mlflow.set_experiment("GAN Training")
    with mlflow.start_run():
        for epoch in range(1000):
            # 1. Train Discriminator
            real_data = get_real_data()
            fake_data = gen_nn(Variable(torch.randn(g_input_size, minibatch_size)))
            d_real_decision = disc_nn(real_data)
            d_fake_decision = disc_nn(fake_data.detach())

            d_real_error = criterion(d_real_decision, Variable(torch.ones(minibatch_size, 1)))  
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(minibatch_size, 1))) 
            d_optimizer.zero_grad()
            d_error = d_real_error + d_fake_error
            d_error.backward()
            d_optimizer.step()

            # 2. Train Generator
            fake_data = gen_nn(Variable(torch.randn(minibatch_size, g_input_size)))
            d_fake_decision = disc_nn(fake_data)
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
        mlflow.pytorch.log_model(gen_nn, "generator")
        mlflow.pytorch.log_model(disc_nn, "discriminator")

train()
