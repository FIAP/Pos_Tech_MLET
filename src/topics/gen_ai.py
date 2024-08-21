import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Configurações principais
image_size = 28  # Tamanho das imagens (MNIST: 28x28)
image_channels = 1  # MNIST tem 1 canal (imagens em escala de cinza)
latent_dim = 2  # Dimensão do espaço latente
batch_size = 128
num_epochs = 20
learning_rate = 0.001

# Definindo transformações para o dataset MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Carregando o dataset MNIST
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Definindo o Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Definindo o Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc2(h))
        return x_reconstructed

# Definindo o VAE que combina Encoder e Decoder
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

# Função de perda para o VAE
def loss_function(x_reconstructed, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Inicializando o modelo, otimizador e a função de perda
input_dim = image_size * image_size
hidden_dim = 256
output_dim = input_dim

encoder = Encoder(input_dim, hidden_dim, latent_dim)
decoder = Decoder(latent_dim, hidden_dim, output_dim)
vae = VAE(encoder, decoder).to(device)

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Treinamento do VAE
def train_vae():
    vae.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, input_dim).to(device)
            optimizer.zero_grad()
            x_reconstructed, mu, logvar = vae(data)
            loss = loss_function(x_reconstructed, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')

    print("Treinamento concluído!")

# Geração de novas imagens
def generate_images(num_images=10):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        generated_images = vae.decoder(z).cpu().view(-1, 1, image_size, image_size)
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(generated_images[i][0], cmap='gray')
            plt.axis('off')
        plt.show()

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Treinando o VAE
train_vae()

# Gerando novas imagens
generate_images()
