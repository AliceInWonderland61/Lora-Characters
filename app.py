"""
Autumn AI Character Chatbot — Clean Button Layout (Gradio 3 Compatible)
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from gtts import gTTS
import tempfile

# -----------------------------
# MODEL + CHARACTER CONFIG
# -----------------------------

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
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, CHARACTERS[character]["adapter"])
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
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    history.append((message, response))

    audio = text_to_speech(response) if enable_tts else None
    return history, audio

# -----------------------------
# COMPREHENSIVE CSS THEME
# -----------------------------

custom_css = """
/* Autumn AI Character Chatbot - Smokey Orange Theme */

/* Main gradient background - Blue-gray to warm cream */
.gradio-container {
    background: linear-gradient(135deg, #8B9DAF, #E8DDD3) !important;
    font-family: 'Georgia', serif;
}

/* Main card styling - Warm cream background */
.main-card {
    max-width: 1100px !important;
    margin: 20px auto !important;
    padding: 25px !important;
    background: rgba(245, 240, 235, 0.95) !important;
    border-radius: 22px !important;
    border: 3px solid #8B6F47 !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25) !important;
}

/* Character buttons - Burnt orange gradient */
.character-btn button {
    width: 100%;
    font-size: 18px !important;
    padding: 14px !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #D97A3A, #C85A28) !important;
    border: 2px solid #8B6F47 !important;
    color: #F5F0EB !important;
    font-weight: 600 !important;
}
.character-btn button:hover {
    background: linear-gradient(135deg, #E68A4A, #D86A38) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
}

/* AGGRESSIVE OVERRIDES FOR ALL GRAY/DARK BACKGROUNDS */

/* Target all possible container elements */
div, section, article, aside, nav, main,
.block, .panel, .form, .container,
[class*="Block"], [class*="block"],
[class*="Container"], [class*="container"],
[data-testid*="block"], [data-testid*="column"] {
    background-color: transparent !important;
}

/* Specifically target the gray rows */
.row, [class*="row"], [data-testid*="row"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Target columns */
.column, [class*="column"], [data-testid*="column"],
.gr-column, [class*="gr-column"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Character button row - Bronze tone background */
.gradio-row, [class*="gradio-row"] {
    background: rgba(139, 111, 71, 0.15) !important;
    padding: 10px !important;
    border-radius: 14px !important;
}

/* Input textbox and textarea - Warm cream */
input, textarea, select,
.input-text, .textbox,
[class*="input"], [class*="textbox"],
input[type="text"], textarea[class*="scroll"] {
    background: #F5F0EB !important;
    background-color: #F5F0EB !important;
    color: #3D3D3D !important;
    border: 2px solid #8B6F47 !important;
}

/* Chatbot container - Dark slate blue */
.chatbot, [data-testid="chatbot"],
.chatbot *, [data-testid="chatbot"] *,
.message-wrap, .message, .message-row,
[class*="chatbot"], [class*="message"] {
    background: #3D4F5C !important;
    background-color: #3D4F5C !important;
    color: #E8DDD3 !important;
}

/* Individual chat messages */
.user-message, .bot-message,
[class*="user"], [class*="bot"],
.message.user, .message.bot {
    background: rgba(139, 111, 71, 0.3) !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

/* Headers and labels - Dark gray */
h1, h2, h3, h4, h5, h6 {
    color: #3D3D3D !important;
    background: transparent !important;
}

label, span, p {
    color: #3D3D3D !important;
}

/* Button styling - Burnt orange */
button {
    background: #D97A3A !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

button:hover {
    background: #C85A28 !important;
}

/* Send button specific styling */
button[variant="primary"] {
    background: #D97A3A !important;
}

/* Audio component */
audio, [class*="audio"] {
    background: #F5F0EB !important;
}

/* Checkbox */
input[type="checkbox"] {
    accent-color: #D97A3A !important;
}

/* Override any remaining dark/gray backgrounds */
.dark, [class*="dark"],
.bg-gray, [class*="bg-gray"],
.bg-slate, [class*="bg-slate"] {
    background: #F5F0EB !important;
    background-color: #F5F0EB !important;
}

/* New Conversation button - Rust/brick color */
.clear-btn button, button:has(svg) {
    background: #A4483D !important;
    color: white !important;
}

.clear-btn button:hover {
    background: #8B3A30 !important;
}

/* Group elements (the boxes around sections) */
.gr-group, [class*="gr-group"],
.group, [class*="Group"] {
    background: rgba(245, 240, 235, 0.95) !important;
    border: 3px solid #8B6F47 !important;
    border-radius: 22px !important;
    padding: 20px !important;
}
"""

# -----------------------------
# UI LAYOUT (buttons + two columns)
# -----------------------------

with gr.Blocks(css=custom_css) as demo:

    # MAIN CARD ---------------------------------------------------
    with gr.Group(elem_classes="main-card"):

        gr.HTML("<h2 style='text-align:center;'>🍂 Choose Your Character</h2>")

        # CHARACTER BUTTONS ROW
        with gr.Row():
            char_buttons = []
            for c in CHARACTERS.keys():
                btn = gr.Button(f"{CHARACTERS[c]['emoji']} {c}", elem_classes="character-btn")
                char_buttons.append(btn)

        # Hidden variable to store selected character
        character_state = gr.State("JARVIS")

        # Link buttons to character state
        for btn, name in zip(char_buttons, CHARACTERS.keys()):
            btn.click(lambda x=name: x, outputs=character_state)

        # TWO-COLUMN CHAT AREA
        with gr.Row():

            # LEFT — message input
            with gr.Column(scale=4):
                msg = gr.Textbox(label="💬 Type your message", lines=3)
                submit_btn = gr.Button("Send", variant="primary")

            # RIGHT — conversation
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(height=400)

    # AUDIO BOX ---------------------------------------------------
    with gr.Group(elem_classes="main-card"):
        enable_tts = gr.Checkbox(label="🔊 Enable Voice", value=True)
        audio_output = gr.Audio(type="filepath", autoplay=True, label="Character Voice Output")

    # RESET BUTTON ------------------------------------------------
    with gr.Group(elem_classes="main-card"):
        clear_btn = gr.Button("🔄 New Conversation")

    # -------------------------
    # Event Logic
    # -------------------------

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

# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)