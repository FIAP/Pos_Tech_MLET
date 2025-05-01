"""
quantization_productization.py

Demonstrates post-training quantization-aware training (QAT) of a Transformer
classification model using PyTorch Lightning, then converts to a real INT8 model
and logs everything with MLflow.
"""

import os
import torch
import torch.nn as nn
import torch.quantization
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.quantization import QuantizationAwareTraining
from torch.utils.data import DataLoader, Dataset
import mlflow.pytorch

# -----------------------------
# 1) Define a LightningModule with QAT support
# -----------------------------
class QATModule(pl.LightningModule):

    def __init__(self,
                 model_name: str = "distilbert-base-uncased",
                 num_labels: int = 2,
                 lr: float = 2e-5):
        """
        - model_name: HuggingFace model to load
        - num_labels: number of classes
        - lr: training learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        # Load a pre-trained DistilBERT for sequence classification
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        # Prepare for quantization-aware training
        self.model.train()  # QAT requires model in train mode
        qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(self.model, qconfig, inplace=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass returns a ModelOutput with .loss and .logits
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)

    def training_step(self, batch, batch_idx):
        # Standard Lightning training step
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("val_loss", outputs.loss, prog_bar=True)

    def configure_optimizers(self):
        # AdamW is typically used for Transformers
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# -----------------------------
# 2) Create Pseudo-Dataset
# -----------------------------
class DummyTextDataset(Dataset):
    def __init__(self, tokenizer, size=500, seq_len=32, num_labels=2):
        """
        Generates random token sequences and random labels.
        """
        texts = ["lorem ipsum"] * size  # dummy placeholder; tokenizer will pad/truncate
        self.encodings = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=seq_len,
            return_tensors="pt"
        )
        # Random labels in [0, num_labels)
        self.labels = torch.randint(0, num_labels, (size,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings['input_ids'][idx],
            "attention_mask": self.encodings['attention_mask'][idx],
            "labels": self.labels[idx]
        }


# -----------------------------
# 3) LightningDataModule for loading data
# -----------------------------
class DummyTextDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=16):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Create a single dummy dataset and split 80/20
        full = DummyTextDataset(self.tokenizer)
        n = len(full)
        split = int(n * 0.8)
        self.train_ds = torch.utils.data.Subset(full, list(range(0, split)))
        self.val_ds = torch.utils.data.Subset(full, list(range(split, n)))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


# -----------------------------
# 4) Main script for training & quantization
# -----------------------------
if __name__ == "__main__":
    # a) Configure MLflow logger
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlf_logger = MLFlowLogger(
        experiment_name="nn_quantization",
        tracking_uri=mlflow.get_tracking_uri()
    )

    # b) Prepare tokenizer, data, and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    data_module = DummyTextDataModule(tokenizer)
    model = QATModule()

    # c) Initialize PyTorch Lightning Trainer with QAT callback
    trainer = pl.Trainer(
        max_epochs=3,
        logger=mlf_logger,
        callbacks=[QuantizationAwareTraining()],
        accelerator="cpu",   # use "gpu" if available
        log_every_n_steps=10
    )

    # d) Train the model with quantization-aware training
    trainer.fit(model, datamodule=data_module)

    # e) Convert to a true quantized model (switches from fake-quant modules to int8)
    quantized_model = torch.quantization.convert(model.model.eval(), inplace=False)

    # f) Save & log the quantized model artifact
    mlflow.pytorch.log_model(
        quantized_model,
        artifact_path="quantized_distilbert",
        registered_model_name="QuantizedDistilBERT"
    )
    print("✅ Quantized model saved and logged to MLflow.")
