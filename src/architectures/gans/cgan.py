import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import mlflow
import mlflow.pytorch


# Define the Conditional Generative Adversarial Network for generator and discriminator (CGAN)


# Data params
data_mean = 4
data_stddev = 1.25

# Model params
g_input_size = 50     # Random noise dimension coming into generator
g_hidden_size = 50    # Generator complexity
g_output_size = 1     # Size of generated output vector
d_input_size = 100    # Minibatch size
d_hidden_size = 50    # Discriminator complexity
d_output_size = 1     # Single dimension for 'real' vs. 'fake' classification
label_size = 10       # Size of label information (e.g., number of classes)
minibatch_size = d_input_size

# Function to generate real data with labels
def get_real_data():
    data = torch.Tensor(np.random.normal(data_mean, data_stddev, (50, minibatch_size)))
    labels = torch.randint(0, label_size, (minibatch_size,))
    return data, labels

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(g_input_size + label_size, g_hidden_size),
            nn.ReLU(),
            nn.Linear(g_hidden_size, g_output_size),
            nn.Sigmoid(),
            nn.Linear(g_input_size + label_size, g_hidden_size),
            nn.Tanh(),
            nn.Linear(g_input_size + label_size, g_hidden_size),
            nn.LeakyReLU(),
            nn.LSTM(g_input_size + label_size, g_hidden_size),
            nn.Tanh()
        )
    
    def forward(self, x, labels):
        # Concatenate label information with noise vector
        x = torch.cat([x, labels], dim=-2)
        return self.model(x)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_input_size + label_size, d_hidden_size),
            nn.ReLU(),
            nn.Linear(d_hidden_size, d_output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        # Concatenate label information with input data
        x = torch.cat([x, labels], dim=-2)
        return self.model(x)

# One-hot encode labels
def one_hot(labels, num_classes):
    return torch.eye(num_classes)[labels]

# Training loop with MLflow logging for CGAN
def train():
    # Initialize models and optimizers
    G = Generator()
    D = Discriminator()
    criterion = nn.BCELoss()
    d_optimizer = Adam(D.parameters(), lr=0.0002)
    g_optimizer = Adam(G.parameters(), lr=0.0002)

    # Start MLflow experiment
    mlflow.set_experiment("CGAN Training")
    with mlflow.start_run():
        for epoch in range(1000):
            # 1. Train Discriminator
            real_data, real_labels = get_real_data()
            real_labels_one_hot = one_hot(real_labels, label_size)
            fake_labels = torch.randint(0, label_size, (minibatch_size,))
            fake_labels_one_hot = one_hot(fake_labels, label_size)
            noise = torch.randn(g_input_size, minibatch_size)
            fake_data = G(Variable(noise), Variable(fake_labels_one_hot))

            d_real_decision = D(real_data, real_labels_one_hot)
            d_fake_decision = D(fake_data.detach(), fake_labels_one_hot)

            d_real_error = criterion(d_real_decision, Variable(torch.ones(minibatch_size, 1)))  
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(minibatch_size, 1))) 
            d_optimizer.zero_grad()
            d_error = d_real_error + d_fake_error
            d_error.backward()
            d_optimizer.step()

            # 2. Train Generator
            fake_data = G(Variable(noise), Variable(fake_labels_one_hot))
            d_fake_decision = D(fake_data, fake_labels_one_hot)
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
        mlflow.log_param("label_size", label_size)

        # Log the final models
        mlflow.pytorch.log_model(G, "generator")
        mlflow.pytorch.log_model(D, "discriminator")

train()
