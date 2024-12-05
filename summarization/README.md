# Summarização

Exemplo de implementação de um modelo de sumarização usando mlflow, bentoml e huggingface.

#### Dev Container

Esse projeto usa dev container para desenvolvimento.

Veja o arquivo `.devcontainer/devcontainer.json` para mais informações sobre o ambiente.

## MLFlow

No terminal para abrir a UI do MLFlow local:
`mlflow ui`.

## Treinamento e Avaliação do Modelo

```
from src.model import HuggingFaceModel
from src.experiment import Experiment

model = HuggingFaceModel()
exp = Experiment(model=model)

# Cria um experimento logando assinatura do modelo, modelo, dependências e o que você quiser adicionar. 
model_info = exp.track(title="Summarization", run_name="default_hugging_face")

# Avalia um modelo de sumarização logado usando o mlflow evaluate em um dataset padrão do hugging face.
exp.evaluate(model_info.model_uri)
```

## Registro de modelo

Quando estamos satisfeitos com a performance de um modelo, podemos registrá-lo para uso no MLFlow Registry, que fará controle de versionamento e de ambiente dos modelos.

```
import mlflow
mlflow.register_model(model_info.model_uri, name="summarization", tags={"status": "demo", "owner": "renata-gotler"})
```

## Deploy usando Flask

Container:
```
docker build --tag=summarization-fast:1.0.0 .
docker run -p 1000:1000 -d summarization-fast:1.0.0
```

## Deploy usando BentoML

```
import bentoml
bento_model = bentoml.mlflow.import_model("summarization", model_info.model_uri)
```

No terminal:
- Visualizar modelos
`bentoml models list`

- Rodar container local
`bentoml serve . --reload`

- Container
`bentoml build --version 1.0.0`
`bentoml containerize summarization:latest`

## Imagens

Listar imagens:
`docker image list`
