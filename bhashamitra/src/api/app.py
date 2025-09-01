# uvicorn src.api.app:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
import random
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    MarianMTModel,
    MarianTokenizer
)

app = FastAPI(title="BhashaMitra API", version="1.0")

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    reply: str
    intent: str
    confidence: float

INTENT_MODEL_PATH = "bhashamitra/models/intent"

intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)

label2id = intent_model.config.label2id
id2label = {v: k for k, v in label2id.items()}

def predict_intent(text):
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = intent_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        conf = probs[0][pred_id].item()
    return id2label[pred_id], conf

chat_model_name = "microsoft/DialoGPT-small"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
chat_model = AutoModelForCausalLM.from_pretrained(chat_model_name)

def generate_chitchat_reply(user_input):
    inputs = chat_tokenizer.encode(user_input + chat_tokenizer.eos_token, return_tensors="pt")
    outputs = chat_model.generate(
        inputs,
        max_length=100,
        pad_token_id=chat_tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    reply = chat_tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return reply

en_hi_model_name = "Helsinki-NLP/opus-mt-en-hi"
hi_en_model_name = "Helsinki-NLP/opus-mt-hi-en"

en_hi_tokenizer = MarianTokenizer.from_pretrained(en_hi_model_name)
en_hi_model = MarianMTModel.from_pretrained(en_hi_model_name)

hi_en_tokenizer = MarianTokenizer.from_pretrained(hi_en_model_name)
hi_en_model = MarianMTModel.from_pretrained(hi_en_model_name)

def translate(text, tokenizer, model):
    if not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def hinglish_to_english(text):
    return translate(text, hi_en_tokenizer, hi_en_model)

def english_to_hinglish(text):
    return translate(text, en_hi_tokenizer, en_hi_model)


fallback_chitchat = [
    "Haan, batao kya chal raha hai? 🙂",
    "Interesting! Aur bolo.",
    "Main yahan hoon, tumhare saath baat karne ke liye 🤗"
]

responses = {
    "greeting": ["Hi! Kaise ho?", "Hello 👋", "Namaste 🙏"],
    "farewell": ["Bye! Milte hain baad mein 👋", "See you later!", "Take care!"],
    "get_weather": ["Kal ka mausam thoda barish ho sakta hai 🌧️", "Aj ka weather clear hai ☀️"],
    "reschedule_meeting": ["Meeting reschedule kar di gayi hai.", "Ok, main meeting postpone kar deta hoon."]
}

def get_response(intent, user_message):
    if intent == "chitchat":
        try:
            # Hinglish → English → DialoGPT → Hinglish
            en_text = hinglish_to_english(user_message)
            if not en_text or en_text == user_message:  # translation failed or unchanged
                return random.choice(fallback_chitchat)

            en_reply = generate_chitchat_reply(en_text)
            if not en_reply.strip():  # DialoGPT failed
                return random.choice(fallback_chitchat)

            reply = english_to_hinglish(en_reply)
            if not reply.strip() or reply == en_reply:  # translation failed
                return random.choice(fallback_chitchat)

            return reply
        except Exception:
            return random.choice(fallback_chitchat)
    else:
        return random.choice(responses.get(intent, ["Samajh nahi aaya 😅"]))

@app.get("/")
def root():
    return {"message": "BhashaMitra API is running!"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    intent, conf = predict_intent(req.text)
    reply = get_response(intent, req.text)
    return ChatResponse(reply=reply, intent=intent, confidence=conf)
