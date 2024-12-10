# Summarização

Exemplo de implementação de um modelo de sumarização usando mlflow, bentoml e huggingface.

#### Dev Container

Esse projeto usa dev container para desenvolvimento.

Veja o arquivo `.devcontainer/devcontainer.json` para mais informações sobre o ambiente.

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