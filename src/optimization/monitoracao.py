import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.optimizer import Optimizer
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def main(version: int, criterion_class: type[_WeightedLoss], optimizer_class: type[Optimizer]):
    # Configurações principais
    start_time = time.time()
    logger.info("Starting job at %s", start_time)
    data_dir = f'{os.getcwd()}\\src\\optimization\\imagens'  # Caminho para o conjunto de dados
    num_classes = 2  # Número de classes para a nova tarefa
    batch_size = 32
    num_epochs = 20
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
    artifact_path = f"{os.getcwd()}/src/optimization/artifacts/experiment_LR_0.001_batchsize_32"
    model = models.resnet18(pretrained=True)
    model_path = os.path.join(artifact_path, "model")

    # Modificando a última camada fully connected para se adequar ao número de classes da nova tarefa
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Definindo o dispositivo (GPU/CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Definindo a função de perda e o otimizador
    criterion = criterion_class()
    optimizer = optimizer_class(model.fc.parameters(), lr=learning_rate)  # type: ignore

    # Iniciando o monitoramento com MLflow
    mlflow.set_experiment("Monitoração em Tempo de Treinamento com ResNet")

    with mlflow.start_run(run_name="Experimento Monitorado 001"):
        mlflow.set_tag("batch_size", batch_size)
        mlflow.set_tag("optimizer", optimizer_class.__name__)
        mlflow.set_tag("criterion", criterion_class.__name__)
        # Logando hiperparâmetros
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)

        # Função de treinamento
        def train_model(model, criterion, optimizer, num_epochs=10):
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
                        logging.info("inputs: %s on phase: %s", inputs, phase)
                        logging.info("labels: %s on phase: %s", labels, phase)
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

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    
                    # Logando métricas no MLflow
                    mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
                    mlflow.log_metric(f"{phase}_acc", epoch_acc.item(), step=epoch)

                    # Exemplo de log de hiperparâmetro durante o treinamento
                    if epoch == 0 and phase == 'train':
                        mlflow.log_param("initial_learning_rate", learning_rate)

            print('Treinamento completo')
            return model

        # Treinando o modelo
        model = train_model(model, criterion, optimizer, num_epochs=num_epochs)

        # Salvando o modelo treinado e logando no MLflow
        mlflow.pytorch.log_model(model, "model")
        torch.save(model.state_dict(), f'{model_path}/model_monitoring_{1}.pth')
        mlflow.log_artifact(model_path, artifact_path)
        end_time = time.time()

        logger.info("Job finished at %s", end_time)
        logger.info("Elapsed time: %s", end_time - start_time)

if __name__ == '__main__':
    main(1, nn.CrossEntropyLoss, optim.Adam)
