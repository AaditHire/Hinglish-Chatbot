import streamlit as st
import requests

st.set_page_config(page_title="BhashaMitra", page_icon="💬", layout="wide")
st.title("💬 BhashaMitra – Hinglish Code-Switching Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Type your message in Hinglish...", key="user_input")

if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "text": user_input})
    # Placeholder bot response
    bot_reply = f"[Bot Response Placeholder] You said: {user_input}"
    st.session_state.messages.append({"role": "bot", "text": bot_reply})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['text']}")
    else:
        st.markdown(f"**🤖 BhashaMitra:** {msg['text']}")
