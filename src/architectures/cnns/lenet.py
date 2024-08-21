import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 20
batch_size = 128
learning_rate = 0.05
num_samples = 10000  # Number of artificial samples to generate
image_size = 32  # Size of the images (32x32 pixels)
num_classes = 10  # Number of classes (e.g., 10 classes for digits 0-9)

# LeNet model
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# Generate artificial data
def generate_artificial_data(num_samples, image_size, num_classes):
    # Generate random images
    images = torch.randn(num_samples, 1, image_size, image_size)
    
    # Generate random labels (integers between 0 and num_classes-1)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    return images, labels

# Create artificial training and testing datasets
train_images, train_labels = generate_artificial_data(
    num_samples,
    image_size,
    num_classes
)

test_images, test_labels = generate_artificial_data(
    num_samples // 10,
    image_size,
    num_classes
)

# Create DataLoaders
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Training the model
def train_model():
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("LeNet Artificial Data Classification")
    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()
                
                # Log metrics every 100 batches
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)
                    mlflow.log_metric("train_accuracy", 100 * correct / total, step=epoch * len(train_loader) + i)

        # Log model parameters
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        # Save the model
        mlflow.pytorch.log_model(model, "lenet_artificial_data_model")

        # Evaluate the model
        evaluate_model(model)

def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    mlflow.log_metric("test_accuracy", accuracy)

# Run the training and evaluation
train_model()
