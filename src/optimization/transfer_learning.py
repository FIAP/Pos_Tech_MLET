import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch


def main():
    # Configurações principais
    data_dir = f'{os.getcwd()}\\src\\optimization\\imagens'  # Caminho para o conjunto de dados
    num_classes = 2  # Número de classes para a nova tarefa
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Transformações nos dados
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Carregando os dados
    image_datasets = {x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Inicializando o modelo ResNet-18 pré-treinado
    model = models.resnet18(pretrained=True)

    # Modificando a última camada fully connected para se adequar ao número de classes da nova tarefa
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Definindo o dispositivo (GPU/CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Definindo a função de perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # Função de treinamento
    def train_model(model, criterion, optimizer, num_epochs=10):
        mlflow.log_params({"num_epochs": num_epochs, "learning_rate": learning_rate})
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Definir o modelo para treinamento
                else:
                    model.eval()  # Definir o modelo para avaliação

                running_loss = 0.0
                running_corrects = 0

                # Iterar sobre os dados
                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Zerar os gradientes
                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward + otimização apenas na fase de treinamento
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Estatísticas
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])

                mlflow.log_metric(f'{phase}_loss', epoch_loss, step=epoch)
                mlflow.log_metric(f'{phase}_acc', epoch_acc.item(), step=epoch)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print('Treinamento completo')
        return model

    # Treinando o modelo
    model = train_model(model, criterion, optimizer, num_epochs=num_epochs)

    # Salvando o modelo treinado
    torch.save(model.state_dict(), 'model_transfer_learning.pth')

if __name__ == '__main__':
    mlflow.set_experiment("Transferencia de Aprendizado com ResNet pré treinada")
    main()