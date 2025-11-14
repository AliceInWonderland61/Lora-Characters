import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

LORA_ADAPTERS = {
    "Jarvis": "AlissenMoreno61/jarvis-lora",
    "Sarcastic": "AlissenMoreno61/sarcastic-lora",
    "Wizard": "AlissenMoreno61/wizard-lora"
}

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

current_adapter = None
current_adapter_name = None

def load_adapter(persona):
    global current_adapter, current_adapter_name
    repo = LORA_ADAPTERS[persona]
    if current_adapter_name != repo:
        current_adapter = PeftModel.from_pretrained(model, repo)
        current_adapter_name = repo

SYSTEM_PROMPTS = {
    "Jarvis": "You respond politely, concisely, professionally and calmly.",
    "Sarcastic": "You respond with heavy sarcasm, witty insults, and an annoyed tone.",
    "Wizard": "You speak like an ancient mystical wizard using magical dramatic language."
}

def chat_fn(message, persona):
    load_adapter(persona)

    # your final fixed prompt
    prompt = (
        f"SYSTEM: {SYSTEM_PROMPTS[persona]}\n"
        f"USER: {message}\n"
        f"ASSISTANT:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = current_adapter.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        stop=["SYSTEM:", "USER:", "ASSISTANT:"]
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # remove prompt leakage
    for tag in ["SYSTEM:", "USER:", "ASSISTANT:"]:
        reply = reply.replace(tag, "")

    return reply.strip()


# ----------- COZY CSS (no white text, NO black bg) -------------
CSS = """
body {
    background: #F6E7D8;
    font-family: 'Georgia', serif;
    color: #5A3E36;
}

/* Chatbot background */
.gr-chatbot {
    background: #F9EFE6 !important;
    border-radius: 18px !important;
    border: 2px solid #D7B7A3 !important;
}

/* Assistant bubble */
.gr-chatbot-message {
    background: #FFE4E8 !important;
    border-radius: 16px !important;
    padding: 12px !important;
    color: #5A3E36 !important;
    border: 1px solid #D9A5A5 !important;
}

/* User bubble */
.gr-chatbot-message.user {
    background: #FCE9D2 !important;
    border: 1px solid #E1C2A3 !important;
}

/* Input box */
textarea {
    background: #FFEAF1 !important;
    color: #5A3E36 !important;
    border-radius: 12px !important;
    border: 1px solid #D9A5A5 !important;
}

/* Buttons */
button {
    background: #D4A373 !important;
    color: white !important;
    border-radius: 14px !important;
    padding: 10px 20px !important;
    border: none !important;
}

"""

# ------------------ UI -------------------
with gr.Blocks(css=CSS) as demo:

    gr.Markdown("<h1 style='text-align:center;'>🍂 Cozy Fall Character Chat 🍁</h1>")

    persona = gr.Radio(
        ["Jarvis", "Sarcastic", "Wizard"],
        value="Jarvis",
        label="Choose Character"
    )

    chatbot = gr.Chatbot(height=420)

    msg = gr.Textbox(placeholder="Type here… 🍁")

    send = gr.Button("Send")

    def respond(user_message, persona, chat_history):
        bot_reply = chat_fn(user_message, persona)
        chat_history.append((user_message, bot_reply))
        return chat_history, ""

    send.click(respond, [msg, persona, chatbot], [chatbot, msg])

demo.launch()
