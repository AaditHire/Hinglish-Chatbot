import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer
from datasets import Dataset

MODEL_DIR = "bhashamitra/models/intent"
TEST_FILE = "bhashamitra/data/processed/intent_test.csv"
LABEL_MAP_FILE = os.path.join(MODEL_DIR, "label2id.json")

def evaluate():
    # Load label map
    with open(LABEL_MAP_FILE, "r") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    # Load test data
    test_df = pd.read_csv(TEST_FILE)
    test_df["label"] = test_df["label"].map(label2id)

    # Tokenizer & model
    tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/indic-bert")
    model = AlbertForSequenceClassification.from_pretrained(MODEL_DIR)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

    test_ds = Dataset.from_pandas(test_df)
    test_ds = test_ds.map(tokenize, batched=True)
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Hugging Face Trainer
    trainer = Trainer(model=model, tokenizer=tokenizer)

    # Predictions
    preds = trainer.predict(test_ds)
    y_true = test_df["label"].values
    y_pred = np.argmax(preds.predictions, axis=1)

    # Report
    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=id2label.values(), yticklabels=id2label.values())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Intent Classification")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    print(f"\n✅ Confusion matrix saved at {MODEL_DIR}/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
