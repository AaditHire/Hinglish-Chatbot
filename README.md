# 💬 BhashaMitra – Hinglish Chatbot

BhashaMitra is a **Hinglish (Hindi + English) conversational chatbot** that can handle both **task-based intents** (like weather, meeting rescheduling, greetings) and **casual chit-chat**.  

It combines:
- ✅ **Intent classification** using **IndicBERT** fine-tuned on a Hinglish dataset  
- ✅ **Task-specific rule-based replies**  
- ✅ **Generative chit-chat** using **DialoGPT**  
- ✅ **Translation pipeline (Hinglish ↔ English)** to make chit-chat responses natural  

---

## 📊 Dataset

This project uses the **[Hinglish Everyday Conversations 1M](https://huggingface.co/datasets/Abhishekcr448/Hinglish-Everyday-Conversations-1M)** dataset from Hugging Face.  

- Contains **~1 million Hinglish conversational pairs**  
- Covers **casual and daily conversation topics**  
- Used for **intent training and chit-chat fine-tuning**  

---

## 🚀 Features
- Hinglish intent detection (`greeting`, `farewell`, `get_weather`, `reschedule_meeting`, `chitchat`)  
- Generative chit-chat with fallback safe replies  
- Streamlit UI for interactive chat  
- FastAPI backend for serving predictions  
- Hinglish-English code-switching support  

---

## 📂 Project Structure
```text
bhashamitra/
├── src/
│   ├── api/                  # FastAPI backend
│   │   └── app.py
│   ├── app/                  # Streamlit frontend
│   │   └── streamlit_chatbot.py
│   ├── chatbot/              # Chat pipeline
│   │   └── pipeline.py
│   ├── data/                 # Data preprocessing
│   ├── generation/           # Chit-chat model
│   ├── models/               # Training + evaluation
│   ├── preprocessing/        # Language utilities
│   └── utils/                # Helpers
├── bhashamitra/models/       # Saved intent model
├── requirements.txt
└── README.md


---

## ⚙️ Installation

Clone this repository:
```bash
git clone https://github.com/your-username/bhashamitra.git
cd bhashamitra
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# OR
source venv/bin/activate      # Linux/Mac

