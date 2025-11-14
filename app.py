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

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

current_adapter_name = None
current_adapter = None


# ---------- Load persona adapter ----------
def ensure_adapter_loaded(persona):
    global current_adapter_name, current_adapter

    adapter_repo = LORA_ADAPTERS[persona]

    if current_adapter_name != adapter_repo:
        print(f"🔄 Switching adapter → {persona} ({adapter_repo})")
        current_adapter = PeftModel.from_pretrained(model, adapter_repo)
        current_adapter_name = adapter_repo


# ---------- Chat Function ----------
def chat(message, persona):
    ensure_adapter_loaded(persona)

    SYSTEM_PROMPTS = {
        "Jarvis": "You are Jarvis: smart, concise, polite, clean answers, no sarcasm.",
        "Sarcastic": "You respond with heavy sarcasm, annoyed tone, witty insults.",
        "Wizard": "You are a wise old fantasy wizard. Speak in mystical, magical style."
    }

    # Construct safe structured prompt
    prompt = (
        f"<s>[SYSTEM]\n{SYSTEM_PROMPTS[persona]}\n[/SYSTEM]\n"
        f"[USER]\n{message}\n[/USER]\n[ASSISTANT]\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = current_adapter.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=False,         # <—— makes ONE ANSWER ONLY
        num_return_sequences=1,
        repetition_penalty=1.1
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)



# ---------- Custom Fall-Themed CSS ----------
FALL_CSS = """
body {
    background: #F4E3D7; /* warm beige */
    font-family: 'Georgia', serif;
}

/* Falling leaves animation */
@keyframes fall {
  0% { transform: translateY(-10vh) rotate(0deg); opacity: 1; }
  100% { transform: translateY(110vh) rotate(360deg); opacity: 0; }
}

.leaf {
    position: fixed;
    top: -10vh;
    font-size: 24px;
    animation: fall linear infinite;
    z-index: 1;
}

/* Chat container */
.gradio-container {
    background-color: transparent !important;
}

/* Pink textboxes */
textarea, input {
    background: #FDECEF !important;
    border-radius: 12px !important;
    border: 1px solid #E7A8A0 !important;
    color: #5A3E36 !important;
    font-size: 16px !important;
}

/* Persona buttons */
button {
    border-radius: 12px !important;
    padding: 10px 16px !important;
}
"""

# ---------- Add Falling Leaves ----------
import random
def falling_leaves_html():
    leaf_emojis = ["🍁", "🍂", "🍃"]
    leaves = ""
    for i in range(18):
        emoji = random.choice(leaf_emojis)
        left = random.randint(0, 100)
        duration = random.uniform(6, 12)
        delay = random.uniform(0, 5)
        leaves += f'<div class="leaf" style="left:{left}vw; animation-duration:{duration}s; animation-delay:{delay}s;">{emoji}</div>'
    return leaves


# ---------- Interface ----------
with gr.Blocks(css=FALL_CSS, head=falling_leaves_html()) as demo:

    gr.Markdown(
        "<h1 style='text-align:center; color:#A65729;'>🍂 Cozy Fall Character Chat 🍁</h1>"
        "<p style='text-align:center;'>Jarvis • Sarcastic • Wizard — cozy autumn vibes, one chat, three personalities.</p>"
    )

    persona = gr.Radio(
        ["Jarvis", "Sarcastic", "Wizard"],
        label="Choose Character",
        value="Jarvis"
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        bubble_full_width=False,
        height=450
    )

    input_box = gr.Textbox(label="Your message", placeholder="Type here... 🍂")
    send_btn = gr.Button("Send")

    def respond(message, persona, history):
        answer = chat(message, persona)
        history.append((message, answer))
        return history, ""

    send_btn.click(
        respond,
        inputs=[input_box, persona, chatbot],
        outputs=[chatbot, input_box]
    )

demo.launch()
