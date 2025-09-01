import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "bhashamitra/models/intent"  # trained model path

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# 🔹 Grab label mapping directly from model config
id2label = model.config.id2label
label2id = model.config.label2id

def predict_intent(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = probs.argmax().item()
        predicted_label = id2label[predicted_class_id]
        return predicted_label, probs[0][predicted_class_id].item()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Kal ka meeting postpone kar do"

    label, confidence = predict_intent(text)
    print(f"📝 Input: {text}")
    print(f"🎯 Predicted Intent: {label} (Confidence: {confidence:.2f})")
