"""
This module contains classes and methods for training LSTM models using PyTorch Lightning.
It includes various training strategies, data pipelines, and model factories.
"""

from typing import override
from dataclasses import dataclass

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from app.model.data import DataPipeline, NoProcessingSingle, NoProcessingMultiple, RangeSingle, RangeMultiple, RangeClusterMultiple
from app.model.lstm import LSTMFactory
from abc import ABC, abstractmethod
from pathlib import Path


@dataclass
class TrainingParams:
    tickers: list[str]
    period: str
    seq_len: int
    num_epochs: int
    learning_rate: float
    batch_size: int
    layer_config: dict
    lstm_params: dict


class LSTMLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training LSTM models.

    Attributes:
        model: The LSTM model to be trained.
        lr: Learning rate for the optimizer.
        criterion: Loss function used for training.
    """

    def __init__(self, model, lr):
        """
        Initialize the LSTMLightningModule.

        Args:
            model: The LSTM model to be trained.
            lr: Learning rate for the optimizer.
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        Perform a forward pass on the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Model predictions.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Perform a training step and log the training loss.

        Args:
            batch: A batch of training data.
            batch_idx: Index of the batch.

        Returns:
            Tensor: Training loss.
        """
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log('train_loss', loss, prog_bar=True)
        # use batch_idx to avoid unused parameter warning
        _ = batch_idx
        return loss

    @override
    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step and log the validation loss.

        Args:
            batch: A batch of validation data.
            batch_idx: Index of the batch.

        Returns:
            Tensor: Validation loss.
        """
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        # use batch_idx to avoid unused parameter warning
        _ = batch_idx
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            Optimizer: Configured optimizer.
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

class TrainingStrategy(ABC):
    """
    Abstract base class for defining training strategies.

    Properties:
        name: Name of the training strategy.

    Methods:
        get_data_pipeline: Returns the data pipeline for the strategy.
        get_model_factory: Returns the model factory for the strategy.
        get_training_params: Returns the training parameters for the strategy.
    """

    def __init__(self, training_params: TrainingParams):
        """
        Initialize the RangeClusterComplexStrategy.

        Args:
            tickers: List of tickers for training.
            period: Training period.
            seq_len: Sequence length for the LSTM.
            num_epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
            batch_size: Batch size for training.
            layer_config: Configuration for LSTM layers.
            lstm_params: Parameters for the LSTM model.
        """
        self.params = dict(
            tickers=training_params.tickers,
            period=training_params.period,
            seq_len=training_params.seq_len,
            num_epochs=training_params.num_epochs,
            learning_rate=training_params.learning_rate,
            batch_size=training_params.batch_size
        )
        self.layer_config = training_params.layer_config
        self.lstm_params = training_params.lstm_params

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the training strategy.

        Returns:
            str: Name of the strategy.
        """

    @abstractmethod
    def get_data_pipeline(self) -> DataPipeline:
        """
        Get the data pipeline for the training strategy.

        Returns:
            DataPipeline: Configured data pipeline.
        """

    @abstractmethod
    def get_model_factory(self) -> LSTMFactory:
        """
        Get the model factory for the training strategy.

        Returns:
            LSTMFactory: Configured model factory.
        """

    @abstractmethod
    def get_training_params(self) -> dict:
        """
        Get the training parameters for the strategy.

        Returns:
            dict: Training parameters.
        """

class NoProcessingSimpleStrategy(TrainingStrategy):
    """
    Training strategy with no feature engineering and simple LSTM configuration.

    Attributes:
        tickers: List of tickers for training.
        period: Training period.
        seq_len: Sequence length for the LSTM.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        batch_size: Batch size for training.
        layer_config: Configuration for LSTM layers.
        lstm_params: Parameters for the LSTM model.
    """

    def __init__(self, training_params: TrainingParams):
        """
        Initialize the NoProcessingSimpleStrategy.

        Args:
            tickers: List of tickers for training.
            period: Training period.
            seq_len: Sequence length for the LSTM.
            num_epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
            batch_size: Batch size for training.
            layer_config: Configuration for LSTM layers.
            lstm_params: Parameters for the LSTM model.
        """
        self._name = "NoProcessingSimple"
        super().__init__(training_params)

    @property
    def name(self):
        """
        Get the name of the training strategy.

        Returns:
            str: Name of the strategy.
        """
        return self._name

    def get_data_pipeline(self) -> DataPipeline:
        """
        Get the data pipeline for the training strategy.

        Returns:
            DataPipeline: Configured data pipeline.
        """
        factory = LSTMFactory(self.layer_config, self.lstm_params)
        model = factory.create()
        return DataPipeline(NoProcessingMultiple(), model, batch_size=self.params['batch_size'])

    def get_model_factory(self) -> LSTMFactory:
        """
        Get the model factory for the training strategy.

        Returns:
            LSTMFactory: Configured model factory.
        """
        return LSTMFactory(self.layer_config, self.lstm_params)

    def get_training_params(self) -> dict:
        """
        Get the training parameters for the strategy.

        Returns:
            dict: Training parameters.
        """
        return self.params

class RangeClusterComplexStrategy(TrainingStrategy):
    """Training with range feature and DBSCAN clustering, using a deeper LSTM configuration."""

    def __init__(self, training_params: TrainingParams):
        """
        Initialize the RangeClusterComplexStrategy.

        Args:
            tickers: List of tickers for training.
            period: Training period.
            seq_len: Sequence length for the LSTM.
            num_epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
            batch_size: Batch size for training.
            layer_config: Configuration for LSTM layers.
            lstm_params: Parameters for the LSTM model.
        """
        self._name = "RangeClusterComplex"
        super().__init__(training_params)

    @property
    def name(self):
        """
        Get the name of the training strategy.

        Returns:
            str: Name of the strategy.
        """
        return self._name

    def get_data_pipeline(self):
        """
        Get the data pipeline for the training strategy.

        Returns:
            DataPipeline: Configured data pipeline.
        """
        factory = LSTMFactory(self.layer_config, self.lstm_params)
        model = factory.create()
        return DataPipeline(RangeClusterMultiple(), model, batch_size=self.params['batch_size'])

    def get_model_factory(self):
        """
        Get the model factory for the training strategy.

        Returns:
            LSTMFactory: Configured model factory.
        """
        return LSTMFactory(self.layer_config, self.lstm_params)

    def get_training_params(self):
        """
        Get the training parameters for the strategy.

        Returns:
            dict: Training parameters.
        """
        return self.params

class NoProcessingSingleStrategy(TrainingStrategy):
    """No feature engineering, single ticker data."""

    def __init__(self, training_params: TrainingParams):
        """
        Initialize the NoProcessingSingleStrategy.

        Args:
            tickers: List of tickers for training.
            period: Training period.
            seq_len: Sequence length for the LSTM.
            num_epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
            batch_size: Batch size for training.
            layer_config: Configuration for LSTM layers.
            lstm_params: Parameters for the LSTM model.
        """
        self._name = "NoProcessingSingle"
        super().__init__(training_params)

    @property
    def name(self): return self._name

    def get_data_pipeline(self):
        """
        Get the data pipeline for the training strategy.

        Returns:
            DataPipeline: Configured data pipeline.
        """
        factory = LSTMFactory(self.layer_config, self.lstm_params)
        model = factory.create()
        return DataPipeline(NoProcessingSingle(), model, batch_size=self.params['batch_size'])

    def get_model_factory(self):
        """
        Get the model factory for the training strategy.

        Returns:
            LSTMFactory: Configured model factory.
        """
        return LSTMFactory(self.layer_config, self.lstm_params)

    def get_training_params(self) -> dict:
        """
        Get the training parameters for the strategy.

        Returns:
            dict: Training parameters.
        """
        return self.params

class NoProcessingMultipleStrategy(TrainingStrategy):
    """No feature engineering, multiple tickers data."""

    def __init__(self, training_params: TrainingParams):
        """
        Initialize the NoProcessingMultipleStrategy.

        Args:
            tickers: List of tickers for training.
            period: Training period.
            seq_len: Sequence length for the LSTM.
            num_epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
            batch_size: Batch size for training.
            layer_config: Configuration for LSTM layers.
            lstm_params: Parameters for the LSTM model.
        """
        self._name = "NoProcessingMultiple"
        super().__init__(training_params)

    @property
    def name(self): return self._name

    def get_data_pipeline(self):
        """
        Get the data pipeline for the training strategy.

        Returns:
            DataPipeline: Configured data pipeline.
        """
        factory = LSTMFactory(self.layer_config, self.lstm_params)
        model = factory.create()
        return DataPipeline(NoProcessingMultiple(), model, batch_size=self.params['batch_size'])

    def get_model_factory(self):
        """
        Get the model factory for the training strategy.

        Returns:
            LSTMFactory: Configured model factory.
        """
        return LSTMFactory(self.layer_config, self.lstm_params)

    def get_training_params(self) -> dict:
        """
        Get the training parameters for the strategy.

        Returns:
            dict: Training parameters.
        """
        return self.params

class RangeSingleStrategy(TrainingStrategy):
    """Daily range feature, single ticker data."""

    def __init__(self, training_params: TrainingParams):
        """
        Initialize the RangeSingleStrategy.

        Args:
            tickers: List of tickers for training.
            period: Training period.
            seq_len: Sequence length for the LSTM.
            num_epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
            batch_size: Batch size for training.
            layer_config: Configuration for LSTM layers.
            lstm_params: Parameters for the LSTM model.
        """
        self._name = "RangeSingle"
        super().__init__(training_params)

    @property
    def name(self): return self._name

    def get_data_pipeline(self):
        """
        Get the data pipeline for the training strategy.

        Returns:
            DataPipeline: Configured data pipeline.
        """
        factory = LSTMFactory(self.layer_config, self.lstm_params)
        model = factory.create()
        return DataPipeline(RangeSingle(), model, batch_size=self.params['batch_size'])

    def get_model_factory(self):
        """
        Get the model factory for the training strategy.

        Returns:
            LSTMFactory: Configured model factory.
        """
        return LSTMFactory(self.layer_config, self.lstm_params)

    def get_training_params(self) -> dict:
        """
        Get the training parameters for the strategy.

        Returns:
            dict: Training parameters.
        """
        return self.params

class RangeMultipleStrategy(TrainingStrategy):
    """Daily range feature, multiple tickers data."""

    def __init__(self, training_params: TrainingParams):
        """
        Initialize the RangeMultipleStrategy.

        Args:
            tickers: List of tickers for training.
            period: Training period.
            seq_len: Sequence length for the LSTM.
            num_epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
            batch_size: Batch size for training.
            layer_config: Configuration for LSTM layers.
            lstm_params: Parameters for the LSTM model.
        """
        self._name = "RangeMultiple"
        super().__init__(training_params)

    @property
    def name(self): return self._name

    def get_data_pipeline(self):
        """
        Get the data pipeline for the training strategy.

        Returns:
            DataPipeline: Configured data pipeline.
        """
        factory = LSTMFactory(self.layer_config, self.lstm_params)
        model = factory.create()
        return DataPipeline(RangeMultiple(), model, batch_size=self.params['batch_size'])

    def get_model_factory(self):
        """
        Get the model factory for the training strategy.

        Returns:
            LSTMFactory: Configured model factory.
        """
        return LSTMFactory(self.layer_config, self.lstm_params)

    def get_training_params(self) -> dict:
        """
        Get the training parameters for the strategy.

        Returns:
            dict: Training parameters.
        """
        return self.params

class RangeClusterMultipleStrategy(TrainingStrategy):
    """Daily range + clustering feature, multiple tickers data."""

    def __init__(self, training_params: TrainingParams):
        """
        Initialize the RangeClusterMultipleStrategy.

        Args:
            tickers: List of tickers for training.
            period: Training period.
            seq_len: Sequence length for the LSTM.
            num_epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
            batch_size: Batch size for training.
            layer_config: Configuration for LSTM layers.
            lstm_params: Parameters for the LSTM model.
        """
        self._name = "RangeClusterMultiple"
        super().__init__(training_params)

    @property
    def name(self): return self._name

    def get_data_pipeline(self):
        """
        Get the data pipeline for the training strategy.

        Returns:
            DataPipeline: Configured data pipeline.
        """
        factory = LSTMFactory(self.layer_config, self.lstm_params)
        model = factory.create()
        return DataPipeline(RangeClusterMultiple(), model, batch_size=self.params['batch_size'])

    def get_model_factory(self):
        """
        Get the model factory for the training strategy.

        Returns:
            LSTMFactory: Configured model factory.
        """
        return LSTMFactory(self.layer_config, self.lstm_params)

    def get_training_params(self) -> dict:
        """
        Get the training parameters for the strategy.

        Returns:
            dict: Training parameters.
        """
        return self.params

class TrainerContext:
    """
    Context for executing training based on a specified strategy.

    Attributes:
        strategy: The training strategy to use.
    """

    def __init__(self, strategy: TrainingStrategy):
        """
        Initialize the TrainerContext.

        Args:
            strategy: The training strategy to use.
        """
        self.strategy = strategy

    def train(self) -> Path:
        """
        Execute the training process using the specified strategy.

        Returns:
            Path: Path to the saved model.
        """
        p = self.strategy.get_training_params()
        pipeline = self.strategy.get_data_pipeline()
        train_loader = pipeline.run(p['tickers'], p['period'], p['seq_len'])
        val_loader = train_loader

        factory = self.strategy.get_model_factory()
        model = factory.create()

        mlf_logger = MLFlowLogger(experiment_name=self.strategy.name, run_name=self.strategy.name)
        trainer = pl.Trainer(
            max_epochs=p['num_epochs'],
            logger=mlf_logger,
            default_root_dir=str(Path(__file__).parent)
        )
        lit_module = LSTMLightningModule(model, p['learning_rate'])
        trainer.fit(lit_module, train_loader, val_loader)

        model_dir = Path(__file__).parent / '.models'
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"{self.strategy.name}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "layer_config": self.strategy.layer_config,
                "lstm_params": self.strategy.lstm_params,
                "strategy": self.strategy.name,
                "training_params": p,
            },
            model_path,
        )
        return model_path
