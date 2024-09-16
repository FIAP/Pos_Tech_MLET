# Configurando uma Feature Store com o Feast

Este guia detalha os passos para configurar uma Feature Store usando o **Feast**, uma ferramenta open-source para gerenciamento de features em projetos de machine learning.

## Índice

- [Pré-requisitos](#pré-requisitos)
- [Passo 1: Instalação do Feast](#passo-1-instalação-do-feast)
- [Passo 2: Inicialização do Repositório de Features](#passo-2-inicialização-do-repositório-de-features)
- [Passo 3: Estrutura do Projeto](#passo-3-estrutura-do-projeto)
- [Passo 4: Configuração do `feature_store.yaml`](#passo-4-configuração-do-feature_storeyaml)
- [Referências](#referências)

---

## Pré-requisitos

- **Python 3.7+** instalado no seu ambiente.
- **PowerShell** disponível para execução de scripts.

## Passo 1: Instalação do Feast

Antes de iniciar, certifique-se de ter um ambiente Python configurado. Recomenda-se o uso de um ambiente virtual.

```powershell
# Crie e ative um ambiente virtual (opcional)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instale o Feast usando pip
pip install feast
```

## Passo 2: Inicialização do Repositório de Features

Crie um diretório para o projeto da Feature Store e inicialize o repositório do Feast.

```powershell
# Crie um diretório para o projeto
New-Item -ItemType Directory -Path "feature_repo"
Set-Location -Path "feature_repo"

# Inicialize o repositório do Feast
feast init
```

## Passo 3: Estrutura do Projeto

Após a inicialização, a estrutura do seu projeto será semelhante a:

```
feature_repo/
├── feature_store.yaml
├── example.py
└── features/
    ├── __init__.py
    └── customer_features.py
```

- **feature_store.yaml**: Arquivo de configuração principal do Feast.
- **example.py**: Exemplo de uso do Feast.
- **features/**: Diretório contendo as definições de features.

## Passo 4: Configuração do `feature_store.yaml`

Edite o arquivo `feature_store.yaml` para configurar o ambiente da Feature Store. Abra o arquivo no editor de sua preferência e ajuste conforme abaixo:

```yaml
project: feature_repo
registry: data/registry.db
provider: local
online_store:
  path: data/online_store.db
```

**Descrição dos campos:**

- **project**: Nome do seu projeto Feast.
- **registry**: Caminho para o registro das features (armazenamento do estado das definições).
- **provider**: Ambiente em que o Feast será executado. Pode ser `local`, `gcp`, `aws`, etc.
- **online_store**: Configurações para o armazenamento online, usado para servir features em tempo real.

## Referências

- [Documentação Oficial do Feast](https://docs.feast.dev/)
- [Repositório no GitHub](https://github.com/feast-dev/feast)

---

# Script PowerShell: Configuração do Repositório

Salve o conteúdo abaixo em um arquivo chamado `setup_feature_store.ps1`.

```powershell
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
```

**Instruções para execução:**

1. Abra o PowerShell.
2. Navegue até o diretório onde o script `setup_feature_store.ps1` está localizado.
3. Execute o script:

```powershell
.\setup_feature_store.ps1
```

---

Este script automatiza os seguintes passos:

- Criação e ativação de um ambiente virtual Python.
- Instalação do Feast.
- Criação do diretório do projeto e inicialização do repositório.
- Criação do diretório de dados.
- Configuração personalizada do arquivo `feature_store.yaml`.

---

**Nota:** Certifique-se de que a execução de scripts está habilitada no PowerShell. Caso não esteja, você pode habilitar temporariamente com o comando:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

---

# Explicação Adicional

- **Ambiente Virtual:** O uso de um ambiente virtual é recomendado para evitar conflitos de dependências entre pacotes Python.
- **Feast:** Uma ferramenta poderosa para gerenciamento de features, permitindo consistência entre o treinamento e a produção de modelos de machine learning.
- **PowerShell Script:** Automatiza a configuração inicial, tornando o processo reproduzível e menos propenso a erros manuais.

---

Agora você tem uma Feature Store básica configurada e pronta para ser expandida nos próximos passos do seu projeto de machine learning.