import pandas as pd
import json
import os

TRAIN_FILE = "bhashamitra/data/processed/intent_train.csv"
OUTPUT_FILE = "bhashamitra/models/intent/label2id.json"

def save_label_map():
    # Load training data
    df = pd.read_csv(TRAIN_FILE)

    # Extract unique labels
    labels = sorted(df["label"].unique())

    # Create mapping
    label2id = {label: idx for idx, label in enumerate(labels)}

    # Ensure output dir exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Save as JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(label2id, f, indent=2)

    print(f"✅ Saved label2id mapping at {OUTPUT_FILE}")

if __name__ == "__main__":
    save_label_map()
