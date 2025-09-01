from datasets import load_dataset
import os

DATA_DIR = "bhashamitra/data/raw/hinglish_convos"
os.makedirs(DATA_DIR, exist_ok=True)

def download_hinglish_convos():
    print("⏳ Loading dataset from Hugging Face...")
    dataset = load_dataset("Abhishekcr448/Hinglish-Everyday-Conversations-1M")
    print("✅ Dataset loaded.")

    # Save a small sample locally for quick experiments
    sample_train = dataset["train"].select(range(10000))
    sample_train.to_csv(os.path.join(DATA_DIR, "sample_train.csv"), index=False)
    print(f"Sample saved at {os.path.join(DATA_DIR, 'sample_train.csv')}")

    return dataset

if __name__ == "__main__":
    ds = download_hinglish_convos()
    print(ds["train"][0])
