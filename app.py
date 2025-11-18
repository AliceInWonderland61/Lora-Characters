"""
Pastel Forest Daisy Character Chatbot
Aesthetic Pastel UI — Soft Green + Soft Rose Pink
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from gtts import gTTS
import tempfile

# ----------------------------
# MODEL SETUP
# ----------------------------
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

CHARACTERS = {
    "JARVIS": {
        "adapter": "AlissenMoreno61/jarvis-lora",
        "emoji": "🌼",
        "description": "Sophisticated AI Assistant",
        "personality": "Professional, articulate, British butler-like"
    },
    "Wizard": {
        "adapter": "AlissenMoreno61/wizard-lora",
        "emoji": "🍃",
        "description": "Mystical Forest Wizard",
        "personality": "Magical, poetic, whimsical"
    },
    "Sarcastic": {
        "adapter": "AlissenMoreno61/sarcastic-lora",
        "emoji": "🌿",
        "description": "Witty and Sharp",
        "personality": "Light sarcasm, playful, still helpful"
    }
}

model_cache = {}

def load_character_model(character):
    if character not in model_cache:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            base_model, CHARACTERS[character]["adapter"]
        )
        model.eval()
        model_cache[character] = model
    return model_cache[character]


def text_to_speech(text, character):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except:
        return None


def chat_with_audio(message, history, character, enable_tts):
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
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        output[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )

    history.append((message, response))
    audio = text_to_speech(response, character) if enable_tts else None

    return history, audio


# ----------------------------
# THEME COLORS
# ----------------------------
PASTEL_FOREST = "#A9C8A6"   # background
SOFT_ROSE = "#F3C2D9"       # buttons
ROSE_BORDER = "#E8BFD1"
OFF_WHITE = "#FFFDFC"
CARD_WHITE = "#FEFEFE"


# ----------------------------
# CUSTOM CSS (Pastel Forest + Soft Rose)
# ----------------------------
custom_css = f"""
.gradio-container {{
    background: {PASTEL_FOREST} !important;
    font-family: 'Quicksand', sans-serif;
}}

/* Centered main wrapper */
.main-card {{
    max-width: 1250px;
    margin: 0 auto;
    padding: 25px;
    border-radius: 22px;
    background: {CARD_WHITE};
    border: 3px solid {ROSE_BORDER};
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}}

.section-card {{
    background: {CARD_WHITE};
    border: 3px solid {ROSE_BORDER};
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}}

button, .gr-button {{
    background: {SOFT_ROSE} !important;
    color: #4A3E4C !important;
    border-radius: 14px !important;
    border: 2px solid {ROSE_BORDER} !important;
    font-weight: 600 !important;
}}

button:hover {{
    background: #F7D0E2 !important;
    transform: translateY(-2px);
}}

input, textarea {{
    background: #fff !important;
    border-radius: 12px !important;
    border: 2px solid {ROSE_BORDER} !important;
}}

#chatbot {{
    border-radius: 18px !important;
    border: 2px solid {ROSE_BORDER} !important;
    background: #ffffff !important;
}}

.character-btn {{
    flex: 1;
    padding: 12px 6px;
    background: {SOFT_ROSE};
    border-radius: 14px;
    border: 2px solid {ROSE_BORDER};
    text-align: center;
    cursor: pointer;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12);
}}

.character-btn:hover {{
    background: #F7D0E2;
}}
"""

# ----------------------------
# UI LAYOUT
# ----------------------------
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.HTML("<h1 style='text-align:center; color:#4A4A4A;'>🌼 Pastel Daisy Assistant 🌼</h1>")

    with gr.Column(elem_classes="main-card"):

        # Character Buttons
        gr.HTML("<h3 style='text-align:center;'>Choose Your Character</h3>")
        with gr.Row():
            char_btns = gr.Radio(
                list(CHARACTERS.keys()),
                value="JARVIS",
                label="",
                elem_classes="character-btn"
            )

        # Chat layout (left: input, right: history)
        with gr.Row():
            with gr.Column(scale=1, elem_classes="section-card"):
                msg = gr.Textbox(
                    placeholder="Type your message…",
                    lines=3
                )
                send_btn = gr.Button("Send")

            with gr.Column(scale=2, elem_classes="section-card"):
                chatbot = gr.Chatbot(height=350, elem_id="chatbot")

        # VOICE SECTION
        with gr.Column(elem_classes="section-card"):
            enable_voice = gr.Checkbox(label="Enable Voice Output", value=False)
            audio_output = gr.Audio(type="filepath")

        gr.Button("New Conversation")


    # LOGIC
    send_btn.click(
        chat_with_audio,
        [msg, chatbot, char_btns, enable_voice],
        [chatbot, audio_output]
    )
    msg.submit(
        chat_with_audio,
        [msg, chatbot, char_btns, enable_voice],
        [chatbot, audio_output]
    )

if __name__ == "__main__":
    demo.launch()
