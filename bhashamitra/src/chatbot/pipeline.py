import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
import json
import random
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from src.generation.chitchat_model import ChitChatGenerator


MODEL_DIR = "bhashamitra/models/intent"

class HinglishChatbot:
    def __init__(self):
        with open(f"{MODEL_DIR}/label2id.json", "r") as f:
            self.label2id = json.load(f)
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/indic-bert")
        self.model = AlbertForSequenceClassification.from_pretrained(MODEL_DIR)
        self.chitchat = ChitChatGenerator()
        self.responses = {
            "greeting": ["Hi! Kaise ho?", "Hello 👋", "Namaste 🙏"],
            "farewell": ["Bye! Milte hain baad mein 👋", "See you later!", "Take care!"],
            "get_weather": ["Kal ka mausam thoda barish ho sakta hai 🌧️", "Aj ka weather clear hai ☀️"],
            "reschedule_meeting": ["Meeting reschedule kar di gayi hai.", "Ok, main meeting postpone kar deta hoon."]
        }

    def predict_intent(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
        return self.id2label[pred_id], confidence
    def get_response(self, text):
        intent, conf = self.predict_intent(text)
        if intent == "chitchat":
            reply, _ = self.chitchat.generate(text)
            return intent, conf, reply
        reply = random.choice(self.responses.get(intent, ["Samajh nahi aaya 🤔"]))
        return intent, conf, reply
if __name__ == "__main__":
    bot = HinglishChatbot()
    while True:
        user_inp = input("📝 You: ")
        intent, conf, reply = bot.get_response(user_inp)
        print(f"🤖 Bot ({intent}, conf={conf:.2f}): {reply}")
