"""
Autumn AI Character Chatbot — Pastel Daisy Theme (Option C) 🌼💚🍑✨
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
    "JARVIS": {"adapter": "AlissenMoreno61/jarvis-lora", "emoji": "🍂"},
    "Wizard": {"adapter": "AlissenMoreno61/wizard-lora", "emoji": "🍁"},
    "Sarcastic": {"adapter": "AlissenMoreno61/sarcastic-lora", "emoji": "🍃"},
}

model_cache = {}

def load_character_model(character):
    if character not in model_cache:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, CHARACTERS[character]["adapter"])
        model.eval()
        model_cache[character] = model
    return model_cache[character]


def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="en", slow=False)
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
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )

    history.append((message, response))

    audio = text_to_speech(response) if enable_tts else None
    return history, audio


# ----------------------------------
# 🌼💚🍑 PASTEL MIX CSS
# ----------------------------------

custom_css = """
/* ---------------------------
   OPTION C — PASTEL DAISY THEME 🌼💚🍑✨
   --------------------------- */

/* Soft pastel background with faint daisy pattern */
.gradio-container {
    background: #DFF2D8 !important; /* pastel green */
    font-family: 'Georgia', serif;
    position: relative;
    overflow-x: hidden;
}

/* Light Daisy Wallpaper */
.gradio-container::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 140%;
    height: 140%;

    background-image:
        radial-gradient(circle at 20% 20%, white 0 35%, transparent 36%),
        radial-gradient(circle at 80% 30%, white 0 40%, transparent 41%),
        radial-gradient(circle at 40% 80%, white 0 33%, transparent 34%),
        radial-gradient(circle at 70% 70%, white 0 38%, transparent 39%);

    mask-image:
        radial-gradient(circle at 20% 20%, transparent 0 30%, black 31%),
        radial-gradient(circle at 80% 30%, transparent 0 35%, black 36%),
        radial-gradient(circle at 40% 80%, transparent 0 28%, black 29%),
        radial-gradient(circle at 70% 70%, transparent 0 33%, black 34%);

    opacity: 0.18;
    z-index: 0;
    animation: drift 30s linear infinite;
}

@keyframes drift {
    0% { transform: translate(0px, 0px) scale(1.2); }
    50% { transform: translate(-40px, 25px) scale(1.25); }
    100% { transform: translate(0px, 0px) scale(1.2); }
}

/* Soft card backgrounds */
.main-card, .gr-group {
    position: relative;
    z-index: 2;
    background: #F7F2E7 !important; /* soft beige */
    border-radius: 22px !important;
    border: 2px solid #EBDCCB !important;
    box-shadow: 0 6px 14px rgba(0,0,0,0.12) !important;
    padding: 25px !important;
    max-width: 1100px !important;
    margin: 20px auto !important;
}

/* Pastel character buttons */
.character-btn button {
    width: 100%;
    font-size: 18px !important;
    padding: 14px !important;
    border-radius: 14px !important;

    background: #FFD97D !important; /* pastel daisy yellow */
    border: 2px solid #F3C55B !important;
    color: #464646 !important;
    font-weight: 600 !important;
}

.character-btn button:hover {
    background: #FFE8AE !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(255, 217, 122, 0.4) !important;
}

/* Buttons */
button {
    background: #FDE8D7 !important; /* pastel peach */
    color: #464646 !important;
    border-radius: 10px !important;
    border: 2px solid #F9C2A5 !important;
    font-weight: 600 !important;
}
button:hover {
    background: #FFEFDF !important;
}

/* Inputs */
input, textarea {
    background: white !important;
    border: 2px solid #F3DCCB !important;
    color: #464646 !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* Chatbox */
.chatbot, [data-testid="chatbot"] {
    background: white !important;
    border-radius: 12px !important;
    border: 2px solid #F3E9D9 !important;
    color: #464646 !important;
}

/* Chat bubbles */
.message.user {
    background: #FDE8D7 !important; /* pastel peach bubble */
    border-radius: 10px !important;
    color: #464646 !important;
}
.message.bot {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    border: 1px solid #F5E6D8 !important;
    color: #464646 !important;
}

/* Headers */
h1, h2, h3, label, p, span {
    color: #464646 !important;
}

/* Checkbox */
input[type="checkbox"] {
    accent-color: #FFD97D !important;
}

/* Audio */
audio {
    background: white !important;
    border-radius: 12px !important;
    border: 2px solid #F3DCCB !important;
}
"""


# ----------------------------------
# UI LAYOUT
# ----------------------------------

with gr.Blocks(css=custom_css) as demo:

    # MAIN CARD
    with gr.Group(elem_classes="main-card"):
        gr.HTML("<h2 style='text-align:center;'>🌼 Choose Your Character</h2>")

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

        with gr.Row():
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    label="💬 Type your message",
                    lines=3,
                    placeholder="Say something..."
                )
                submit_btn = gr.Button("Send", variant="primary")

            with gr.Column(scale=6):
                chatbot = gr.Chatbot(height=400)

    # AUDIO PANEL
    with gr.Group(elem_classes="main-card"):
        enable_tts = gr.Checkbox(label="🔊 Enable Voice", value=True)
        audio_output = gr.Audio(type="filepath", autoplay=True, label="Character Voice Output")

    # RESET
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
# RUN
# ----------------------------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
