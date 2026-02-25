# Módulo `model` — Arquitetura LSTM e Pipeline de Dados

Este módulo contém a definição do modelo LSTM, sua fábrica (Factory Method), os hiperparâmetros validados e o pipeline de dados baseado no padrão Strategy.

---

## Estrutura de Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `lstm_params.py` | Modelo Pydantic com os hiperparâmetros da LSTM |
| `lstm.py` | Classe `LSTM` (nn.Module) e `LSTMFactory` |
| `data.py` | Estratégias de ingestão e processamento de dados |
| `__init__.py` | Exportações do pacote |

---

## Visão Geral da Arquitetura

```mermaid
classDiagram
    class LSTMParams {
        +int input_size
        +int hidden_size
        +int num_layers
        +int output_size
        +bool batch_first
    }

    class LSTMFactory {
        +dict layer_config
        +LSTMParams params
        +get_layer(layer_name, input_size) nn.Module
        +create() LSTM
    }

    class LSTM {
        +nn.ModuleList layers
        +forward(x) Tensor
    }

    class DataStrategy {
        <<abstract>>
        +process(tickers, period, seq_len)*
        +load_data(tickers, period)$ DataFrame
        +create_sequences(data, seq_len)$ tuple
    }

    class DataPipeline {
        +DataStrategy strategy
        +LSTM model
        +int batch_size
        +run(tickers, period, seq_len) DataLoader
    }

    DataStrategy <|-- NoProcessingSingle
    DataStrategy <|-- NoProcessingMultiple
    DataStrategy <|-- RangeSingle
    DataStrategy <|-- RangeMultiple
    DataStrategy <|-- RangeClusterMultiple

    LSTMFactory --> LSTMParams
    LSTMFactory --> LSTM
    DataPipeline --> DataStrategy
    DataPipeline --> LSTM
```

---

## Passo-a-Passo: Como Usar

### 1. Definir Hiperparâmetros

```python
from app.model.lstm_params import LSTMParams

params = LSTMParams(
    input_size=4,       # High, Low, Close, Volume
    hidden_size=64,
    num_layers=2,
    output_size=1,      # Preço de fechamento previsto
    batch_first=True,
)
```

### 2. Construir o Modelo com a Factory

```python
from app.model.lstm import LSTMFactory

layer_config = {
    "lstm1": "LSTM",
    "linear1": "Linear",
    "sigmoid1": "Sigmoid",
}

factory = LSTMFactory(layer_config, params)
model = factory.create()
print(model)
```

O fluxo interno da factory é:

```mermaid
flowchart LR
    A[layer_config dict] --> B[LSTMFactory.create]
    B --> C{Para cada layer_type}
    C -->|LSTM| D[nn.LSTM]
    C -->|Linear| E[nn.Linear]
    C -->|Sigmoid| F[nn.Sigmoid]
    C -->|Softmax| G[nn.Softmax]
    D & E & F & G --> H[LSTM model]
```

### 3. Escolher uma Estratégia de Dados

| Estratégia | Tickers | Feature Engineering | Clustering |
|---|---|---|---|
| `NoProcessingSingle` | 1 | Nenhum | Não |
| `NoProcessingMultiple` | N | Nenhum | Não |
| `RangeSingle` | 1 | High - Low | Não |
| `RangeMultiple` | N | High - Low | Não |
| `RangeClusterMultiple` | N | High - Low | DBSCAN |

### 4. Executar o Pipeline de Dados

```python
from app.model.data import DataPipeline, RangeMultiple

strategy = RangeMultiple()
pipeline = DataPipeline(strategy, model, batch_size=32)

loader = pipeline.run(
    tickers=["AAPL", "MSFT"],
    period="1y",
    seq_len=30,
)

for X_batch, y_batch in loader:
    print(X_batch.shape, y_batch.shape)
    break
```

```mermaid
sequenceDiagram
    participant User
    participant Pipeline as DataPipeline
    participant Strategy as DataStrategy
    participant YF as yfinance
    participant MLflow

    User->>Pipeline: run(tickers, period, seq_len)
    Pipeline->>MLflow: start_run + log_param
    Pipeline->>Strategy: process(tickers, period, seq_len)
    Strategy->>YF: Ticker(t).history(period)
    YF-->>Strategy: DataFrame
    Strategy->>Strategy: feature engineering + create_sequences
    Strategy-->>Pipeline: (X, y) tensors
    Pipeline->>Pipeline: TensorDataset → DataLoader
    Pipeline-->>User: DataLoader
```

---

## Camadas Suportadas pela Factory

| Nome | Classe PyTorch | Parâmetros de entrada |
|---|---|---|
| `LSTM` | `nn.LSTM` | `input_size`, `hidden_size`, `num_layers`, `batch_first` |
| `Linear` | `nn.Linear` | `input_size` (dinâmico), `output_size` |
| `Sigmoid` | `nn.Sigmoid` | — |
| `Softmax` | `nn.Softmax(dim=1)` | — |

Caso um `layer_name` não reconhecido seja fornecido, a factory levanta `ValueError`.

---

## Testes Relacionados

- `tests/test_lstm_factory.py` — Instanciação, forward pass, camadas inválidas.
- `tests/test_lstm_data.py` — Estratégias de dados com yfinance mockado.

Execute com:

```bash
pytest tests/test_lstm_factory.py tests/test_lstm_data.py -v
```
