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
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# We will store loaded adapters to avoid re-loading every time
loaded_adapters = {}

def load_adapter(persona):
    """Load adapter once and cache it."""
    repo = LORA_ADAPTERS[persona]

    if persona not in loaded_adapters:
        print(f"Loading LoRA adapter for: {persona}")
        loaded_adapters[persona] = PeftModel.from_pretrained(base_model, repo)
        loaded_adapters[persona].eval()

    return loaded_adapters[persona]


def format_prompt(message, persona):
    """Ensures the LoRA knows how to respond in character."""
    return f"User: {message}\n{persona}:"


def chat(message, persona):
    adapter_model = load_adapter(persona)

    prompt = format_prompt(message, persona)

    inputs = tokenizer(prompt, return_tensors="pt").to(adapter_model.device)

    outputs = adapter_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.8,
        do_sample=True,
    )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant’s response
    if f"{persona}:" in full:
        reply = full.split(f"{persona}:")[-1].strip()
    else:
        reply = full.strip()

    return reply


# ---------------- UI ----------------
with gr.Blocks() as iface:
    gr.Markdown("# 🧠 Multi-Persona LLaMA (Jarvis, Sarcastic, Wizard)")
    persona = gr.Dropdown(["Jarvis", "Sarcastic", "Wizard"], label="Persona", value="Jarvis")
    inp = gr.Textbox(label="Message")
    out = gr.Textbox(label="Response")
    btn = gr.Button("Send")
    btn.click(chat, inputs=[inp, persona], outputs=out)

iface.launch()
