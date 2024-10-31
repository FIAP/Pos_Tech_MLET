import torch

import torch.nn as nn
import torch.nn.utils.prune as prune


# Definindo uma rede neural simples
class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Inicializando a rede neural
model = SimpleNN()

# Exemplo de poda por magnitude
# Mantém apenas os pesos mais significativos (maiores em valor absoluto)
prune.l1_unstructured(model.fc1, name='weight', amount=0.2)  # Poda 20% dos pesos

# Exemplo de poda por estrutura
# Poda canais inteiros (útil para redes convolucionais)
prune.ln_structured(model.fc2, name='weight', amount=0.3, n=2, dim=0)  # Poda 30% dos canais

# Exemplo de poda de neurônios
# Poda neurônios inteiros (útil para redes densas)
class NeuronPruning(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        neuron_importance = t.abs().sum(dim=1)
        threshold = torch.quantile(neuron_importance, 0.2)
        mask[neuron_importance < threshold, :] = 0
        return mask

prune.global_unstructured(
    [(model.fc1, 'weight'), (model.fc2, 'weight')],
    pruning_method=NeuronPruning,
    amount=0.2
)

# Exemplo de poda global
# Poda pesos em todas as camadas de forma global
parameters_to_prune = (
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)
prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)

# Removendo as máscaras de poda (opcional, para tornar a poda permanente)
for module, param in parameters_to_prune:
    prune.remove(module, param)

# Exibindo os pesos podados
print("Pesos da camada fc1 após poda:")
print(model.fc1.weight)
print("Pesos da camada fc2 após poda:")
print(model.fc2.weight)
print("Pesos da camada fc3 após poda:")
print(model.fc3.weight)
