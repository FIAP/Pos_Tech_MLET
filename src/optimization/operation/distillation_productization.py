"""
distillation_productization.py

Performs knowledge distillation from a large “teacher” MLP to a smaller “student” MLP,
using PyTorch Lightning, and logs the student model to MLflow.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import MLFlowLogger
import mlflow.pytorch

# -----------------------------
# 1) Define LightningModule for Distillation
# -----------------------------
class DistillModule(pl.LightningModule):
    def __init__(self,
                 teacher: nn.Module,
                 student: nn.Module,
                 lr: float = 1e-3,
                 temperature: float = 5.0,
                 alpha: float = 0.5):
        """
        teacher: pretrained large model
        student: smaller model to be trained
        temperature: softening factor for logits
        alpha: weight between distillation loss and hard-label loss
        """
        super().__init__()
        # Only log hyperparams, not full modules
        self.save_hyperparameters(ignore=['teacher', 'student'])
        self.teacher = teacher.eval()  # freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.student = student

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Teacher inference (no gradient)
        with torch.no_grad():
            t_logits = self.teacher(x)
        # Student inference
        s_logits = self.student(x)

        # 1) Distillation loss (KL between softened distributions)
        T = self.hparams.temperature
        t_probs = F.softmax(t_logits / T, dim=1)
        s_log_probs = F.log_softmax(s_logits / T, dim=1)
        distill_loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (T * T)

        # 2) Hard-label cross-entropy
        task_loss = F.cross_entropy(s_logits, y)

        # Combined loss
        loss = self.hparams.alpha * distill_loss + (1 - self.hparams.alpha) * task_loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Only student parameters are optimized
        return torch.optim.Adam(self.student.parameters(), lr=self.hparams.lr)


# -----------------------------
# 2) Pseudo-DataLoader
# -----------------------------
def make_dummy_loader(num_samples=2000,
                      input_dim=20,
                      num_classes=2,
                      batch_size=64):
    """
    Returns a DataLoader with random features and labels.
    """
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# -----------------------------
# 3) Assemble teacher & student
# -----------------------------
# Teacher: larger 2-layer network
teacher_model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
# Simulate teacher pre-training
with torch.no_grad():
    # A few dummy forward passes to initialize
    _ = teacher_model(torch.randn(5, 20))

# Student: smaller network
student_model = nn.Sequential(
    nn.Linear(20, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)


# -----------------------------
# 4) Main script for distillation
# -----------------------------
if __name__ == "__main__":
    # a) Configure MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns_distill"))
    mlf_logger = MLFlowLogger(
        experiment_name="nn_distillation",
        tracking_uri=mlflow.get_tracking_uri()
    )

    # b) Prepare Lightning module and data
    distill_module = DistillModule(
        teacher=teacher_model,
        student=student_model,
        lr=1e-3,
        temperature=5.0,
        alpha=0.7
    )
    train_loader = make_dummy_loader()

    # c) Trainer setup
    trainer = pl.Trainer(
        max_epochs=10,
        logger=mlf_logger,
        accelerator="cpu",
        log_every_n_steps=20
    )

    # d) Run distillation training
    trainer.fit(distill_module, train_loader)

    # e) After training, log the trained student model
    mlflow.pytorch.log_model(
        distill_module.student,
        artifact_path="distilled_student",
        registered_model_name="DistilledMLPStudent"
    )
    print("✅ Distilled student model saved and logged to MLflow.")
