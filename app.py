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

    prompt = message
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = current_adapter.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- HTML HEADER WITH GOOGLE FONTS + LEAVES ---
HEADER_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500&family=Poppins:wght@400;600&family=Cormorant+Garamond:wght@500;700&display=swap" rel="stylesheet">

<style>
    body { background: #FFF2EB !important; }

    #falling-leaves {
        pointer-events: none;
        position: fixed;
        top: 0; left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        z-index: 0;
    }
    .leaf {
        position: absolute;
        width: 30px;
        opacity: 0.7;
        animation: fall linear infinite;
    }
    @keyframes fall {
        0% { transform: translateY(-10vh) rotate(0deg); }
        100% { transform: translateY(110vh) rotate(360deg); }
    }

    .jarvis-text { font-family: 'Playfair Display', serif !important; }
    .sarcastic-text { font-family: 'Poppins', sans-serif !important; }
    .wizard-text { font-family: 'Cormorant Garamond', serif !important; }

    .jarvis-glow {
        border: 2px solid #D7C4B7;
        box-shadow: 0 0 12px rgba(255, 255, 255, 0.6);
        border-radius: 8px;
        padding: 10px;
    }

    .sarcastic-shake {
        animation: shake 0.2s ease-in-out;
    }
    @keyframes shake {
        0% { transform: translateX(0); }
        25% { transform: translateX(-3px); }
        50% { transform: translateX(3px); }
        75% { transform: translateX(-2px); }
        100% { transform: translateX(0); }
    }

    .wizard-embers {
        background-image: radial-gradient(circle, rgba(255,200,150,0.4) 2px, transparent 2px);
        background-size: 6px 6px;
        animation: embers 3s linear infinite;
    }
    @keyframes embers {
        from { background-position: 0 0; }
        to { background-position: 0 -100px; }
    }
</style>

<div id="falling-leaves"></div>

<script>
const leafContainer = document.getElementById("falling-leaves");
const leafImgs = [
    "https://i.imgur.com/bR9P7Pf.png",
    "https://i.imgur.com/1W1O1Gy.png",
    "https://i.imgur.com/NK7TnZq.png"
];

function spawnLeaf() {
    const leaf = document.createElement("img");
    leaf.src = leafImgs[Math.floor(Math.random() * leafImgs.length)];
    leaf.classList.add("leaf");
    leaf.style.left = Math.random() * 100 + "vw";
    leaf.style.animationDuration = (5 + Math.random() * 7) + "s";
    leaf.style.opacity = 0.5 + Math.random() * 0.5;
    leaf.style.width = 20 + Math.random() * 25 + "px";
    leafContainer.appendChild(leaf);

    setTimeout(() => leaf.remove(), 12000);
}
setInterval(spawnLeaf, 500);

// JS persona updater
function updatePersona(selected) {
    const chatBox = document.querySelector(".gr-textbox");
    if (!chatBox) return;

    chatBox.classList.remove("jarvis-text","sarcastic-text","wizard-text");

    if (selected === "Jarvis") chatBox.classList.add("jarvis-text");
    if (selected === "Sarcastic") chatBox.classList.add("sarcastic-text","sarcastic-shake");
    if (selected === "Wizard") chatBox.classList.add("wizard-text","wizard-embers");
}
</script>
"""

with gr.Blocks(css="body {background:#FFF2EB;}") as ui:
    gr.HTML(HEADER_HTML)

    persona = gr.Radio(
        ["Jarvis","Sarcastic","Wizard"],
        label="Choose Character",
        value="Jarvis"
    )

    hidden_js = gr.Textbox(visible=False)

    def pass_signal(p):
        return p

    persona.change(pass_signal, inputs=persona, outputs=hidden_js, _js="updatePersona")

    chatbox = gr.Chatbot(height=350)
    msg = gr.Textbox(label="Your message", placeholder="Type here… 🍁")

    send_btn = gr.Button("Send")

    send_btn.click(chat_fn, inputs=[msg, persona], outputs=chatbox)

ui.launch()
