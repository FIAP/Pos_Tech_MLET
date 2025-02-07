import os
import mlflow
import numpy as np
import evaluate

from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

def main(accuracy_threshold=0.9):
    # 1) Load the IMDB dataset
    dataset = load_dataset("imdb")
    # Splits: {"train": 25000 examples, "test": 25000 examples}

    # 2) Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Preprocessing function: tokenize text
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    # 3) Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # Remove original text column to keep only input_ids, attention_mask, labels
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["test"]

    # 4) Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2  # IMDB: 2 classes (positive/negative)
    )

    # 5) Define metric & function for computing metrics
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # 6) TrainingArguments - set up your hyperparams, output dirs, etc.
    training_args = TrainingArguments(
        output_dir="distilbert-imdb-checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True
    )

    # 7) Create a Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # ------------------------- MLflow Integration -------------------------
    # We'll start an MLflow run and log everything inside it.

    with mlflow.start_run(run_name="DistilBERT-IMDB-finetune") as run:
        # Log hyperparameters (some come from TrainingArguments)
        mlflow.log_param("learning_rate", training_args.learning_rate)
        mlflow.log_param("epochs", training_args.num_train_epochs)
        mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
        mlflow.log_param("weight_decay", training_args.weight_decay)

        # 8) Train (fine-tune) the model
        trainer.train()

        # 9) Evaluate on test set
        metrics = trainer.evaluate(test_dataset)
        print("Evaluation metrics:", metrics)
        # Log metrics (accuracy, etc.) to MLflow
        mlflow.log_metrics({
            "eval_loss": metrics["eval_loss"],
            "eval_accuracy": metrics["eval_accuracy"]
        })

        # 10) (Optional) Save model & tokenizer (as artifacts in MLflow)
        # We'll first save them locally, then log the folder as an artifact
        save_dir = "distilbert-imdb-mlflow"
        trainer.save_model(save_dir)         # HF's save_model
        tokenizer.save_pretrained(save_dir)  # save tokenizer

        # Log the entire directory as an MLflow artifact
        if metrics["eval_accuracy"] >= accuracy_threshold:
            mlflow.log_artifacts(save_dir, artifact_path="model_files")

        # 11) Let's do an example inference with the final model
        # We'll load the local saved model to ensure it works
        from transformers import DistilBertForSequenceClassification
        loaded_model = DistilBertForSequenceClassification.from_pretrained(save_dir)

        test_texts = [
            "This movie was awesome! The acting was great and the plot was so exciting.",
            "I didn't like this film. It was incredibly slow and the story was boring."
        ]
        inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = loaded_model(**inputs)
        pred_labels = outputs.logits.argmax(dim=-1)
        print("Test texts:", test_texts)
        print("Predicted labels:", pred_labels.tolist())
        # 1 => positive, 0 => negative on IMDB

    # End of MLflow run scope. This commits all logs, metrics, artifacts.

if __name__ == "__main__":
    main()
