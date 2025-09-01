from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ChitChatGenerator:
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, user_input, history=[]):
        # Encode new input
        new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt")

        # Append to history
        bot_input_ids = torch.cat([torch.tensor(history), new_input_ids], dim=-1) if history else new_input_ids

        # Generate reply
        output_ids = self.model.generate(
            bot_input_ids,
            max_length=200,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        reply = self.tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return reply, output_ids.tolist()
