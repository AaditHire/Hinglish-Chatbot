# 💬 BhashaMitra – Hinglish Chatbot

BhashaMitra is a **Hinglish (Hindi + English) conversational chatbot** that can handle both **task-based intents** (like weather, meeting rescheduling, greetings) and **casual chit-chat**.

It combines:
- ✅ **Intent classification** using **IndicBERT** fine-tuned on a Hinglish dataset  
- ✅ **Task-specific rule-based replies**  
- ✅ **Generative chitchat** using **DialoGPT**  
- ✅ **Translation pipeline (Hinglish ↔ English)** to make chit-chat responses natural  

---

## 🚀 Features
- Hinglish intent detection (`greeting`, `farewell`, `get_weather`, `reschedule_meeting`, `chitchat`)  
- Generative chit-chat with fallback safe replies  
- Streamlit UI for interactive chat  
- FastAPI backend for serving predictions  
- Hinglish-English code-switching support  

---

## 📂 Project Structure
bhashamitra/
├── src/
│ ├── api/ # FastAPI backend
│ │ └── app.py
│ ├── app/ # Streamlit frontend
│ │ └── streamlit_chatbot.py
│ ├── chatbot/ # Chat pipeline
│ │ └── pipeline.py
│ ├── data/ # Data preprocessing
│ ├── generation/ # Chit-chat model
│ ├── models/ # Training + evaluation
│ ├── preprocessing/ # Language utilities
│ └── utils/ # Helpers
├── bhashamitra/models/ # Saved intent model
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Installation

Clone this repository:
```bash
git clone https://github.com/your-username/bhashamitra.git
cd bhashamitra
Create a virtual environment:

bash
Copy code
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# OR
source venv/bin/activate      # Linux/Mac
Install dependencies:

bash
Copy code
pip install -r requirements.txt
▶️ Usage
Start the FastAPI backend
bash
Copy code
uvicorn src.api.app:app --reload
Runs at: http://127.0.0.1:8000

Start the Streamlit frontend
bash
Copy code
streamlit run src/app/streamlit_chatbot.py
Open in browser: http://localhost:8501

🧠 Training Intent Classifier
To retrain the intent model:

bash
Copy code
python src/models/intent_trainer.py
Evaluate:

bash
Copy code
python src/models/evaluate_intent.py

