import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

# Configurações principais
image_size = 64  # Você pode ajustar este valor conforme o tamanho das suas imagens
image_channels = 3  # Assumindo que suas imagens são RGB (3 canais)
g_input_size = 100  # Dimensão do ruído aleatório para o gerador
g_hidden_size = 256  # Complexidade do gerador
g_output_size = image_channels * image_size * image_size  # Tamanho de saída do gerador
d_hidden_size = 256  # Complexidade do discriminador
d_output_size = 1  # Saída única para 'real' vs 'fake'
batch_size = 64
num_epochs = 50
learning_rate = 0.0002

# Caminho para os dados (atualize para o seu caminho de imagens)
data_dir = f'{os.getcwd()}\\src\\optimization\\imagens'

# Definindo transformações para o dataset
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(image_channels)], [0.5 for _ in range(image_channels)])
])

# Carregando os dados de treino e validação
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Generator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_fn, dropout_rate):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size * 2),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_fn, dropout_rate):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size * 4),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Função de treinamento com MLflow logging
def train(activation_fn, l2_lambda, dropout_rate, version):
    G = Generator(g_input_size, g_hidden_size, g_output_size, activation_fn, dropout_rate).to(device)
    D = Discriminator(g_output_size, d_hidden_size, d_output_size, activation_fn, dropout_rate).to(device)

    criterion = nn.BCELoss()
    d_optimizer = Adam(D.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    g_optimizer = Adam(G.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    experiment_name = f"GAN Custom Dataset Experiment {version}"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"GAN_{activation_fn.__name__}_L2_{l2_lambda}_Dropout_{dropout_rate}_v{version}"):
        mlflow.set_tags(
            {   
                "version": version,
                "activation_fn": activation_fn.__name__,
                "l2_lambda": l2_lambda,
                "dropout_rate": dropout_rate
            }
        )
        mlflow.log_param("g_input_size", g_input_size)
        mlflow.log_param("g_hidden_size", g_hidden_size)
        mlflow.log_param("g_output_size", g_output_size)
        mlflow.log_param("d_hidden_size", d_hidden_size)
        mlflow.log_param("d_output_size", d_output_size)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(num_epochs):
            for i, (images, _) in enumerate(train_loader):
                images = images.view(-1, g_output_size).to(device)
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)

                outputs = D(images)
                d_loss_real = criterion(outputs, real_labels)
                real_score = outputs

                z = torch.randn(batch_size, g_input_size).to(device)
                fake_images = G(z)
                outputs = D(fake_images)
                d_loss_fake = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = d_loss_real + d_loss_fake
                D.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                z = torch.randn(batch_size, g_input_size).to(device)
                fake_images = G(z)
                outputs = D(fake_images)
                g_loss = criterion(outputs, real_labels)

                G.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
                mlflow.log_metric("d_loss", d_loss.item(), step=epoch)
                mlflow.log_metric("g_loss", g_loss.item(), step=epoch)
        
        #input_example = torch.randn(1, g_input_size).to(device)  # Exemplo de entrada para o Gerador
        mlflow.pytorch.log_model(G, f"generator_v{version}")
        mlflow.pytorch.log_model(D, f"discriminator_v{version}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activation_functions = [nn.ReLU, nn.Sigmoid, nn.Tanh]
l2_lambdas = [0.0, 0.01, 0.1]
dropout_rates = [0.0, 0.3, 0.5]

def main(version):
    for activation_fn in activation_functions:
        for l2_lambda in l2_lambdas:
            for dropout_rate in dropout_rates:
                print(f"Running experiment with {activation_fn.__name__}, L2: {l2_lambda}, Dropout: {dropout_rate}, Version: {version}")
                train(activation_fn, l2_lambda, dropout_rate, version)

if __name__ == "__main__":
    version = "1.0"
    main(version)
