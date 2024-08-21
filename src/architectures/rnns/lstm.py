import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mlflow
import mlflow.pytorch

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 10      # Number of features in the input data
hidden_size = 50     # Number of hidden units in the LSTM
num_layers = 2       # Number of LSTM layers
output_size = 1      # Number of output units (e.g., regression output)
num_epochs = 50
batch_size = 64
learning_rate = 0.001
sequence_length = 20  # Length of the input sequences
num_samples = 10000  # Number of artificial samples to generate

# Generate artificial data
def generate_artificial_data(num_samples, sequence_length, input_size):
    # Generate random sequences of data
    X = torch.randn(num_samples, sequence_length, input_size)
    
    # Generate random labels (regression target)
    y = torch.randn(num_samples, 1)
    
    return X, y

# Create artificial training and testing datasets
train_X, train_y = generate_artificial_data(num_samples, sequence_length, input_size)
test_X, test_y = generate_artificial_data(num_samples // 10, sequence_length, input_size)

# Create DataLoaders
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# LSTM model
class LSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            output_size
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid(output_size, input_size)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm1(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.sigmoid(out[:, -1, :])
        out, _ = self.lstm2(x, (h0, c0))
        out = self.fc2(out[:, -1, :])
        return out

# Training the model
def train_model():
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("LSTM Artificial Data Regression")
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("intermediate_networks", [
            model.lstm1.__name__,
            model.fc1.__name__,
            model.sigmoid.__name__
        ])
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("output_size", output_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for i, (sequences, labels) in enumerate(train_loader):
                sequences, labels = sequences.to(device), labels.to(device)

                # Forward pass
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                # Log metrics every 100 batches
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)

        # Save the model
        mlflow.pytorch.log_model(model, "lstm_artificial_data_model")

        # Evaluate the model
        evaluate_model(model, criterion)

def evaluate_model(model, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss:.4f}")
    mlflow.log_metric("test_loss", average_test_loss)

# Run the training and evaluation
train_model()
