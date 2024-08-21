import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer, BertConfig
import numpy as np
import mlflow
import mlflow.pytorch

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 768  # BERT's hidden size
hidden_size = 50  # Custom hidden layer size
num_classes = 2  # Number of output classes
num_epochs = 5
batch_size = 32
learning_rate = 1e-4
sequence_length = 20  # Length of the input sequences
num_samples = 1000  # Number of artificial samples to generate

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
bert = BertModel.from_pretrained('bert-base-uncased', config=config).to(device)

# Generate artificial data
def generate_artificial_data(num_samples, sequence_length, num_classes):
    # Generate random text data
    sentences = [" ".join(["word"] * sequence_length) for _ in range(num_samples)]
    
    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Tokenize the sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=sequence_length, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    return input_ids, attention_mask, labels

# Create artificial training and testing datasets
train_input_ids, train_attention_mask, train_labels = generate_artificial_data(num_samples, sequence_length, num_classes)
test_input_ids, test_attention_mask, test_labels = generate_artificial_data(num_samples // 10, sequence_length, num_classes)

# Create DataLoaders
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# BERT-based model
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size, num_classes):
        super().__init__()
        self.bert = bert
        self.fc = nn.Linear(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = outputs[0][:, 0, :]  # Take the output of the [CLS] token

        x = self.relu(self.fc(hidden_state))
        x = self.classifier(x)
        return x

# Training the model
def train_model():
    model = BERTClassifier(bert, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("BERT Artificial Data Classification")
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Log metrics every 100 batches
                if i % 100 == 0:
                    accuracy = 100 * correct / total
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)
                    mlflow.log_metric("train_accuracy", accuracy, step=epoch * len(train_loader) + i)

        # Save the model
        mlflow.pytorch.log_model(model, "bert_artificial_data_model")

        # Evaluate the model
        evaluate_model(model, criterion)

def evaluate_model(model, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    mlflow.log_metric("test_loss", average_test_loss)
    mlflow.log_metric("test_accuracy", accuracy)

# Run the training and evaluation
train_model()
