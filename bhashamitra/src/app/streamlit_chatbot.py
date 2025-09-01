import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="BhashaMitra Chatbot", page_icon="💬", layout="centered")

st.title("💬 BhashaMitra Chatbot")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input
user_input = st.text_input("✍️ Type your message:", "")

if st.button("Send") and user_input.strip():
    try:
        # Send request to FastAPI backend
        response = requests.post(API_URL, json={"text": user_input})
        data = response.json()

        reply = data.get("reply", "Samajh nahi aaya 😅")
        intent = data.get("intent", "unknown")
        confidence = data.get("confidence", 0.0)

        # Save to history
        st.session_state["messages"].append(
            {"role": "user", "content": user_input}
        )
        st.session_state["messages"].append(
            {"role": "bot", "content": reply, "intent": intent, "confidence": confidence}
        )

    except Exception as e:
        st.error(f"API Error: {e}")

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **Bot:** {msg['content']}")
        st.caption(f"🎯 Intent: `{msg['intent']}` | 📊 Confidence: `{msg['confidence']:.2f}`")
