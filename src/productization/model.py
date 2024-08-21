import io
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(64 * 64 * 3, 2)  # Exemplo de modelo simples

    def forward(self, x):
        return self.fc(x)

# Função para carregar o modelo
def load_model(model_path):
    model = SimpleModel()
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
