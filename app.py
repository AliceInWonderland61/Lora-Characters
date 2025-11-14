import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

LORA_ADAPTERS = {
    "Jarvis": "AlissenMoreno61/jarvis-lora",
    "Sarcastic": "AlissenMoreno61/sarcastic-lora",
    "Wizard": "AlissenMoreno61/wizard-lora"
}

print("Loading base model…")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def load_adapter(name):
    print(f"Loading adapter: {name}")
    return PeftModel.from_pretrained(model, LORA_ADAPTERS[name])

current_adapter = load_adapter("Jarvis")

def chat_fn(message, persona):
    global current_adapter
    current_adapter = load_adapter(persona)

    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    outputs = current_adapter.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )

    # Return only the last assistant message
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------ HTML + CSS + JS ------------------

HEADER_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500&family=Poppins:wght@300;600&family=Cormorant+Garamond:wght@500;700&display=swap" rel="stylesheet">

<style>
    body {
        background: #FFE9E3 !important;
        font-family: 'Poppins', sans-serif;
    }

    #falling-leaves {
        pointer-events: none;
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        overflow: hidden;
        z-index: 0;
    }

    .leaf {
        position: absolute;
        width: 35px;
        opacity: 0.8;
        animation: fall linear infinite;
    }

    @keyframes fall {
        0%   { transform: translateY(-10vh) rotate(0deg); }
        100% { transform: translateY(110vh) rotate(360deg); }
    }

    /* Persona fonts */
    .jarvis-text { font-family: 'Playfair Display', serif !important; }
    .sarcastic-text { font-family: 'Poppins', sans-serif !important; }
    .wizard-text { font-family: 'Cormorant Garamond', serif !important; }

    /* Persona Effects */
    .sarcastic-shake {
        animation: shake 0.25s ease-in-out;
    }
    @keyframes shake {
        0% { transform: translateX(0); }
        25% { transform: translateX(-2px); }
        50% { transform: translateX(2px); }
        75% { transform: translateX(-1px); }
        100% { transform: translateX(0); }
    }

    .wizard-embers {
        background-image: radial-gradient(circle, rgba(255,200,150,0.4) 2px, transparent 2px);
        background-size: 6px 6px;
        animation: embers 3s linear infinite;
    }
    @keyframes embers {
        from { background-position: 0 0; }
        to   { background-position: 0 -120px; }
    }
</style>

<div id="falling-leaves"></div>

<script>
function spawnLeaf() {
    const leafContainer = document.getElementById("falling-leaves");
    const leaf = document.createElement("img");

    const files = [
        "file=leaves/leaf1.png",
        "file=leaves/leaf2.png",
        "file=leaves/leaf3.png",
        "file=leaves/leaf4.png"
    ];

    leaf.src = files[Math.floor(Math.random()*files.length)];
    leaf.classList.add("leaf");
    leaf.style.left = Math.random()*100 + "vw";
    leaf.style.animationDuration = (5 + Math.random()*6) + "s";
    leaf.style.width = (25 + Math.random()*20) + "px";

    leafContainer.appendChild(leaf);

    setTimeout(() => leaf.remove(), 12000);
}
setInterval(spawnLeaf, 500);

// Called from Python
function updatePersona(p) {
    const chat = document.querySelector(".gr-textbox");

    if (!chat) return;

    chat.classList.remove("jarvis-text","sarcastic-text","wizard-text",
                           "sarcastic-shake","wizard-embers");

    if (p === "Jarvis")  chat.classList.add("jarvis-text");
    if (p === "Sarcastic") chat.classList.add("sarcastic-text","sarcastic-shake");
    if (p === "Wizard") chat.classList.add("wizard-text","wizard-embers");
}
</script>
"""

# -----------------------------------------------------

with gr.Blocks() as demo:
    gr.HTML(HEADER_HTML)

    persona = gr.Radio(["Jarvis","Sarcastic","Wizard"],
                        label="Choose Character", value="Jarvis")

    chatbox = gr.Chatbot(height=420)
    text = gr.Textbox(label="Your message", placeholder="Type here… 🍁")

    send_btn = gr.Button("Send")

    # Hook to run JS updatePersona()
    js_call = gr.HTML("", visible=False)

    def trigger_js(p):
        return gr.update(value=f"<script>updatePersona('{p}')</script>")

    persona.change(trigger_js, inputs=persona, outputs=js_call)

    send_btn.click(chat_fn, inputs=[text, persona], outputs=chatbox)

demo.launch()
