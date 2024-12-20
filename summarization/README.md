# Summarização

Exemplo de implementação de um modelo de sumarização usando mlflow, bentoml e huggingface.

## Set Up

- Você precisará ter instalado em sua máquina o [Docker](https://www.docker.com/) e o [Make](https://www.gnu.org/software/make/).

### Ambiente

- Criação de um ambiente virtual.
```
python -m venv fiap_mlflow_bentoml
fiap_mlflow_bentoml\Scripts\Activate.ps1
```
- Instalação dos pacotes necessários
```make install-dev```

## MLFlow

No terminal, rode o comando abaixo para abrir a UI do MLFlow localmente.
`mlflow ui`.

[Documentação MLFlow](https://mlflow.org/)

## Experimentação e Registro do Melhor Model

No terminal rode o comando:
`make select-model`

## Deploy usando BentoML

No terminal:
- Visualizar modelos registrados BentoML
`make list-models`

- Rodar modelo registrado localmente
`make serve`

- Gerar container
`make build`
`make containerize`

[Documentação BentoML](https://www.bentoml.com/)

## Teste de extresse no endpoint

No terminal:
`make load-test`

[Documentação locust](https://locust.io/)