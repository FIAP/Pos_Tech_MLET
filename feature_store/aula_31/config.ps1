# Crie e ative um ambiente virtual (opcional)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instale o Feast
pip install feast

# Crie o diretório do projeto
New-Item -ItemType Directory -Path "feature_repo"
Set-Location -Path "feature_repo"

# Inicialize o repositório do Feast
feast init

# Crie o diretório para os dados
New-Item -ItemType Directory -Path "data"

# Configuração personalizada do feature_store.yaml
$featureStoreConfig = @"
project: feature_repo
registry: data/registry.db
provider: local
online_store:
  path: data/online_store.db
"@

# Salve a configuração no arquivo feature_store.yaml
Set-Content -Path "feature_store.yaml" -Value $featureStoreConfig

Write-Host "Configuração da Feature Store concluída com sucesso."
