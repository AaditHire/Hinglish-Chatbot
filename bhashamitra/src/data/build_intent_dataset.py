from datasets import load_dataset
import pandas as pd
import os

INTENT_RULES = {
    "greeting": ["hello", "hi", "haan", "namaste", "good morning", "good evening"],
    "farewell": ["bye", "good night", "alvida", "milte", "take care"],
    "get_weather": ["weather", "mausam", "barish", "rain", "garmi", "temperature"],
    "reschedule_meeting": ["meeting", "shift", "reschedule", "cancel", "postpone"],
    "chitchat": ["kya kar", "chal raha", "khel", "movie", "netflix", "pubg"],
}

DATA_DIR = "bhashamitra/data/processed"

def assign_intent(text):
    text = text.lower()
    for intent, keywords in INTENT_RULES.items():
        for kw in keywords:
            if kw in text:
                return intent
    return None  # ignore if no match

def build_intent_dataset():
    ds = load_dataset("Abhishekcr448/Hinglish-Everyday-Conversations-1M", split="train")
    df = pd.DataFrame(ds)

    # Apply labeling
    df["label"] = df["input"].apply(assign_intent)

    # Drop rows with no intent
    df = df.dropna(subset=["label"])
    df = df[["input", "label"]].rename(columns={"input": "text"})

    os.makedirs(DATA_DIR, exist_ok=True)

    # 🔹 Save full dataset
    train_full = df.sample(frac=0.8, random_state=42)
    test_full = df.drop(train_full.index)
    train_full.to_csv(f"{DATA_DIR}/intent_train_full.csv", index=False)
    test_full.to_csv(f"{DATA_DIR}/intent_test_full.csv", index=False)
    print(f"✅ Saved FULL dataset: {len(train_full)} train, {len(test_full)} test")

    # 🔹 Save mini dataset (fast training)
    train_mini = train_full.sample(n=20000, random_state=42)
    test_mini = test_full.sample(n=5000, random_state=42)
    train_mini.to_csv(f"{DATA_DIR}/intent_train.csv", index=False)
    test_mini.to_csv(f"{DATA_DIR}/intent_test.csv", index=False)
    print(f"✅ Saved MINI dataset: {len(train_mini)} train, {len(test_mini)} test")

if __name__ == "__main__":
    build_intent_dataset()
