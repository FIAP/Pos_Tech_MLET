import io
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Função para carregar o modelo
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Função de inferência
def predict(model, image):
    logger.info("Image bytes: %s", io.BytesIO(image))
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)  # Adicionar dimensão para o batch
    logger.info("Image shape: %s", image.shape)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()
