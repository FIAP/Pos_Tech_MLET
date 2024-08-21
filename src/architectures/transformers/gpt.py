import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import numpy as np
import mlflow
import mlflow.pytorch

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 768  # GPT's hidden size
num_epochs = 5
batch_size = 32
learning_rate = 1e-4
sequence_length = 20  # Length of the input sequences
num_samples = 1000  # Number of artificial samples to generate

# Load pre-trained GPT model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
gpt = GPT2LMHeadModel.from_pretrained('gpt2', config=config).to(device)

# Generate artificial data
def generate_artificial_data(num_samples, sequence_length):
    # Generate random text data
    sentences = [" ".join(["word"] * sequence_length) for _ in range(num_samples)]
    
    # Tokenize the sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=sequence_length, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    return input_ids, attention_mask

# Create artificial training and testing datasets
train_input_ids, train_attention_mask = generate_artificial_data(num_samples, sequence_length)
test_input_ids, test_attention_mask = generate_artificial_data(num_samples // 10, sequence_length)

# Create DataLoaders
train_dataset = TensorDataset(train_input_ids, train_attention_mask)
test_dataset = TensorDataset(test_input_ids, test_attention_mask)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# GPT-based model
class GPTModel(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt1 = gpt
        self.gpt2 = gpt

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        return outputs.loss, outputs.logits

# Training the model
def train_model():
    model = GPTModel(gpt).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("GPT Artificial Data Generation")
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for i, (input_ids, attention_mask) in enumerate(train_loader):
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

                # Forward pass
                loss, _ = model(input_ids, attention_mask)

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
        mlflow.pytorch.log_model(model, "gpt_artificial_data_model")

        # Evaluate the model
        evaluate_model(model)

def evaluate_model(model):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask in test_loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            loss, _ = model(input_ids, attention_mask)
            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss:.4f}")
    mlflow.log_metric("test_loss", average_test_loss)

# Run the training and evaluation
train_model()
