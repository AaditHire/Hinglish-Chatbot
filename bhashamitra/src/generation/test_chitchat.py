from src.generation.chitchat_model import ChitChatGenerator

if __name__ == "__main__":
    chat = ChitChatGenerator()

    while True:
        user_inp = input("📝 You: ")
        if user_inp.lower() in ["exit", "quit", "bye"]:
            print("👋 Exiting test mode...")
            break

        reply, conf = chat.generate(user_inp)
        print(f"🤖 ChitChatBot: {reply}")
