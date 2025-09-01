import pandas as pd
from transformers import (
    AlbertTokenizer, 
    AlbertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers.trainer_callback import EarlyStoppingCallback

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

def train_intent_model(train_file, test_file, output_dir):
    # Load data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Encode labels
    labels = sorted(train_df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    train_df["label_id"] = train_df["label"].map(label2id)
    test_df["label_id"] = test_df["label"].map(label2id)

    # Convert to Dataset
    train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label_id"]])

    # Load tokenizer and model
    model_name = "ai4bharat/indic-bert"
    tokenizer = AlbertTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AlbertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(labels), 
        id2label=id2label, 
        label2id=label2id
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        num_train_epochs=5,   # can try 3-5
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ Model saved at {output_dir}")

if __name__ == "__main__":
    train_intent_model(
        "bhashamitra/data/processed/intent_train.csv", 
        "bhashamitra/data/processed/intent_test.csv", 
        "bhashamitra/models/intent"
    )
