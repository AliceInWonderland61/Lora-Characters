"""
Autumn AI Character Chatbot — Light Autumn Air Theme 🍂✨
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from gtts import gTTS
import tempfile

# ----------------------------------
# MODEL + CHARACTERS
# ----------------------------------

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

CHARACTERS = {
    "JARVIS": {
        "adapter": "AlissenMoreno61/jarvis-lora",
        "emoji": "🍂",
    },
    "Wizard": {
        "adapter": "AlissenMoreno61/wizard-lora",
        "emoji": "🍁",
    },
    "Sarcastic": {
        "adapter": "AlissenMoreno61/sarcastic-lora",
        "emoji": "🍃",
    },
}

model_cache = {}

def load_character_model(character):
    if character not in model_cache:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, CHARACTERS[character]["adapter"])
        model.eval()
        model_cache[character] = model
    return model_cache[character]


def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except:
        return None


def chat_fn(message, history, character, enable_tts):
    if not message.strip():
        return history, None

    model = load_character_model(character)

    messages = []
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
    )

    history.append((message, response))

    audio = text_to_speech(response) if enable_tts else None
    return history, audio


# ----------------------------------
# LIGHT AUTUMN AIR CSS 🍂
# ----------------------------------

custom_css = """
/* ---------------------------
   LIGHT AUTUMN AIR THEME 🍂🍁
   Soft, cozy, airy fall colors
   --------------------------- */
.gradio-container {
    background: linear-gradient(135deg, #8A9BB3 0%, #F7E9D5 40%, #FAF4EC 100%) !important;
    font-family: 'Georgia', serif;
}

/* Warm card containers */
.main-card, .gr-group {
    max-width: 1100px !important;
    margin: 20px auto !important;
    padding: 25px !important;
    background: rgba(250, 244, 236, 0.96) !important;
    border-radius: 22px !important;
    border: 3px solid #9A6A3A !important;
    box-shadow: 0 6px 18px rgba(70, 83, 102, 0.25) !important;
}

/* Character buttons */
.character-btn button {
    width: 100%;
    font-size: 18px !important;
    padding: 14px !important;
    border-radius: 14px !important;

    background: linear-gradient(135deg, #D57400, #E18E34) !important;
    border: 2px solid #A53A1A !important;
    color: #FAF4EC !important;
    font-weight: 600 !important;
}
.character-btn button:hover {
    background: linear-gradient(135deg, #E89C40, #F0A857) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(165, 58, 26, 0.25) !important;
}

/* Standard buttons */
button {
    background: #D57400 !important;
    border-radius: 10px !important;
    border: 2px solid #A53A1A !important;
    color: #FAF4EC !important;
    font-weight: 600 !important;
}
button:hover {
    background: #E18E34 !important;
}

/* Inputs */
input, textarea {
    background: #FAF4EC !important;
    border: 2px solid #9A6A3A !important;
    color: #465366 !important;
    border-radius: 12px !important;
}

/* Chatbot area */
.chatbot, [data-testid="chatbot"] {
    background: #F2E8DC !important;
    color: #3A4654 !important;
    border-radius: 12px !important;
    border: 2px solid #CFA97B !important;
}

/* Chat messages */
.message.user {
    background: #E0C9A6 !important;
    color: #3E4654 !important;
    border-radius: 10px !important;
}
.message.bot {
    background: #F6EAD9 !important;
    border-radius: 10px !important;
    color: #3E4654 !important;
}

/* Text + Headers */
h1, h2, h3, label, p, span {
    color: #465366 !important;
}

/* Checkbox */
input[type="checkbox"] {
    accent-color: #D57400 !important;
}

/* Audio */
audio {
    background: #FAF4EC !important;
    border-radius: 12px !important;
    border: 2px solid #A53A1A !important;
}
"""


# ----------------------------------
# UI LAYOUT
# ----------------------------------

with gr.Blocks(css=custom_css) as demo:

    # MAIN CARD -------------------------------
    with gr.Group(elem_classes="main-card"):

        gr.HTML("<h2 style='text-align:center;'>🍂 Choose Your Character</h2>")

        # Character Buttons
        with gr.Row():
            char_buttons = []
            for name in CHARACTERS.keys():
                b = gr.Button(
                    f"{CHARACTERS[name]['emoji']} {name}",
                    elem_classes="character-btn"
                )
                char_buttons.append(b)

        character_state = gr.State("JARVIS")

        for btn, name in zip(char_buttons, CHARACTERS.keys()):
            btn.click(lambda x=name: x, outputs=character_state)

        # Chat columns (side by side)
        with gr.Row():

            # LEFT: Message input
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    label="💬 Type your message",
                    lines=3,
                    placeholder="Say something..."
                )
                submit_btn = gr.Button("Send", variant="primary")

            # RIGHT: Chat
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(height=400)

    # AUDIO PANEL -----------------------------
    with gr.Group(elem_classes="main-card"):
        enable_tts = gr.Checkbox(label="🔊 Enable Voice", value=True)
        audio_output = gr.Audio(type="filepath", autoplay=True, label="Character Voice Output")

    # RESET -----------------------------------
    with gr.Group(elem_classes="main-card"):
        clear_btn = gr.Button("🔄 New Conversation")

    # Logic
    submit_btn.click(
        chat_fn,
        inputs=[msg, chatbot, character_state, enable_tts],
        outputs=[chatbot, audio_output],
    ).then(lambda: "", outputs=msg)

    msg.submit(
        chat_fn,
        inputs=[msg, chatbot, character_state, enable_tts],
        outputs=[chatbot, audio_output],
    ).then(lambda: "", outputs=msg)

    clear_btn.click(lambda: ([], None), outputs=[chatbot, audio_output])


# ----------------------------------
# LAUNCH
# ----------------------------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
