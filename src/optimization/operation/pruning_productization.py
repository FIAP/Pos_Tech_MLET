"""
pruning_productization.py

Demonstrates iterative magnitude-based pruning on a simple feed-forward network
using PyTorch Lightning's ModelPruning callback and logs the final sparse model.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset
import mlflow.pytorch

# -----------------------------
# 1) Define a LightningModule to prune
# -----------------------------
class PruningModel(pl.LightningModule):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        # Simple 2-layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# -----------------------------
# 2) Generate Pseudo-Data
# -----------------------------
def make_dummy_dataloader(num_samples=1000, input_dim=128, num_classes=10, batch_size=32):
    """
    Creates a DataLoader with random features and random labels.
    """
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

# -----------------------------
# 3) Main script for pruning
# -----------------------------
if __name__ == "__main__":
    # a) Setup MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns_pruning"))
    mlf_logger = MLFlowLogger(
        experiment_name="nn_pruning",
        tracking_uri=mlflow.get_tracking_uri()
    )

    # b) Instantiate model, dataloader
    model = PruningModel()
    train_loader = make_dummy_dataloader()

    # c) Configure Trainer with ModelPruning callback for 50% global sparsity
    prune_callback = ModelPruning(
        method="l1_unstructured",   # magnitude‐based unstructured pruning
        amount=0.5                  # target 50% of all weights
    )
    trainer = pl.Trainer(
        max_epochs=5,
        logger=mlf_logger,
        callbacks=[prune_callback],
        accelerator="cpu"
    )

    # d) Run training (pruning happens gradually under the hood)
    trainer.fit(model, train_loader)

    # e) Finalize pruning: remove pruning reparameterization to hard-zero weights
    for module in [model.fc1, model.fc2]:
        prune.remove(module, 'weight')

    # f) Log the pruned model
    mlflow.pytorch.log_model(
        model,
        artifact_path="pruned_mlp",
        registered_model_name="PrunedMLP"
    )
    print("✅ Pruned model saved and logged to MLflow.")
