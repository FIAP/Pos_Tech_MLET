import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import mlflow
import mlflow.pytorch
import numpy as np

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 3
batch_size = 16
learning_rate = 1e-4
sequence_length = 20  # Length of the input sequences
num_samples = 500  # Number of artificial samples to generate

# Load pre-trained T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
config = T5Config.from_pretrained('t5-small')
t5 = T5ForConditionalGeneration.from_pretrained('t5-small', config=config).to(device)

# Generate artificial data
def generate_artificial_data(num_samples, sequence_length):
    # Generate random text data for input and target
    input_sentences = ["translate English to French: " + " ".join(["word"] * sequence_length) for _ in range(num_samples)]
    target_sentences = ["mot " * sequence_length for _ in range(num_samples)]

    # Tokenize the sentences
    input_encodings = tokenizer(input_sentences, padding=True, truncation=True, max_length=sequence_length, return_tensors="pt")
    target_encodings = tokenizer(target_sentences, padding=True, truncation=True, max_length=sequence_length, return_tensors="pt")

    input_ids = input_encodings['input_ids']
    attention_mask = input_encodings['attention_mask']
    target_ids = target_encodings['input_ids']

    return input_ids, attention_mask, target_ids

# Create artificial training and testing datasets
train_input_ids, train_attention_mask, train_target_ids = generate_artificial_data(num_samples, sequence_length)
test_input_ids, test_attention_mask, test_target_ids = generate_artificial_data(num_samples // 10, sequence_length)

# Create DataLoaders
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_target_ids)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_target_ids)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# T5-based model
class T5Model(nn.Module):
    def __init__(self, t5):
        super().__init__()
        self.t5 = t5

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

# Training the model
def train_model():
    model = T5Model(t5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("T5 Artificial Data Generation")
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for i, (input_ids, attention_mask, target_ids) in enumerate(train_loader):
                input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(device)

                # Forward pass
                loss, _ = model(input_ids, attention_mask, target_ids)

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
        mlflow.pytorch.log_model(model, "t5_artificial_data_model")

        # Evaluate the model
        evaluate_model(model)

def evaluate_model(model):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, target_ids in test_loader:
            input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(device)
            loss, _ = model(input_ids, attention_mask, target_ids)
            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss:.4f}")
    mlflow.log_metric("test_loss", average_test_loss)

# Run the training and evaluation
train_model()
