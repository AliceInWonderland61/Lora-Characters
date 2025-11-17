"""
Fall-Themed Character Chatbot with Text-to-Speech
REDESIGNED - Beautiful Centered Layout
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from gtts import gTTS
import tempfile

# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

CHARACTERS = {
    "JARVIS": {
        "adapter": "AlissenMoreno61/jarvis-lora",
        "emoji": "🍂",
        "description": "Sophisticated AI Assistant",
        "personality": "Professional, articulate, British butler-like"
    },
    "Wizard": {
        "adapter": "AlissenMoreno61/wizard-lora",
        "emoji": "🍁",
        "description": "Mystical Sage of Autumn",
        "personality": "Poetic, uses medieval language, mystical"
    },
    "Sarcastic": {
        "adapter": "AlissenMoreno61/sarcastic-lora",
        "emoji": "🍃",
        "description": "Witty & Sharp-Tongued",
        "personality": "Ryan Reynolds wit, cheeky but helpful"
    }
}

model_cache = {}

def load_character_model(character):
    if character not in model_cache:
        print(f"Loading {character}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, CHARACTERS[character]["adapter"])
        model.eval()
        model_cache[character] = model
        print(f"✅ {character} loaded!")
    return model_cache[character]

def text_to_speech(text, character):
    """Convert character's response to speech - FAST"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def chat_with_audio(message, history, character, enable_tts):
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
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    history.append((message, response))
    
    audio_file = None
    if enable_tts:
        audio_file = text_to_speech(response, character)
    
    return history, audio_file

# ============================================================================
# REDESIGNED BEAUTIFUL INTERFACE
# ============================================================================

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #8B9DC3 0%, #C49A6C 30%, #DFB77B 60%, #E67E22 100%) !important;
    font-family: 'Georgia', 'Times New Roman', serif;
}

@keyframes fall {
    0% { transform: translateY(-10vh) rotate(0deg); opacity: 1; }
    100% { transform: translateY(110vh) rotate(720deg); opacity: 0.3; }
}

@keyframes sway {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-20px); }
    75% { transform: translateX(20px); }
}

.leaf {
    position: fixed;
    top: -10vh;
    z-index: 1;
    pointer-events: none;
    animation: fall linear infinite, sway ease-in-out infinite;
    filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3));
}

footer { display: none !important; }

/* Character Selection - Horizontal */
#character-radio {
    display: flex !important;
    justify-content: center !important;
    gap: 15px !important;
}

#character-radio label {
    background: rgba(255, 248, 220, 0.95) !important;
    border: 3px solid #8B4513 !important;
    border-radius: 20px !important;
    padding: 20px 30px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    color: #5D4037 !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    min-width: 150px !important;
    text-align: center !important;
}

#character-radio label:hover {
    transform: translateY(-5px) scale(1.05) !important;
    box-shadow: 0 8px 25px rgba(139, 69, 19, 0.4) !important;
}

#character-radio input:checked + label {
    background: linear-gradient(135deg, #FFE4B5, #DEB887) !important;
    color: #5D4037 !important;
    border-color: #CD853F !important;
    box-shadow: 0 10px 30px rgba(205, 133, 63, 0.6) !important;
    transform: translateY(-8px) scale(1.1) !important;
}

#character-radio input:checked + label::after {
    content: ' ✨';
}

#chatbot {
    border-radius: 25px !important;
    border: 4px solid #8B4513 !important;
    background: rgba(255, 250, 245, 0.98) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

.message.user {
    background: linear-gradient(135deg, #FFB347, #FF8C42) !important;
    color: white !important;
    border-radius: 20px 20px 5px 20px !important;
}

.message.bot {
    background: rgba(255, 248, 220, 0.95) !important;
    border: 2px solid #DEB887 !important;
    border-radius: 20px 20px 20px 5px !important;
    color: #2C1810 !important;
    font-weight: 500 !important;
}

.message.bot p, .message.user p {
    color: inherit !important;
}

textarea {
    border: 3px solid #8B4513 !important;
    border-radius: 20px !important;
    background: rgba(255, 250, 245, 0.98) !important;
    font-size: 16px !important;
    color: #5D4037 !important;
}

textarea:focus {
    border-color: #FFB347 !important;
    box-shadow: 0 0 15px rgba(255, 179, 71, 0.5) !important;
}

button.primary {
    background: linear-gradient(135deg, #FFB347, #FF8C42) !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 15px 35px !important;
    font-weight: bold !important;
    font-size: 18px !important;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #FFA500, #FF7F00) !important;
    transform: translateY(-3px) scale(1.05) !important;
    box-shadow: 0 8px 25px rgba(255, 127, 0, 0.5) !important;
}

button.secondary {
    background: rgba(255, 248, 220, 0.9) !important;
    color: #8B4513 !important;
    border: 2px solid #CD853F !important;
    border-radius: 20px !important;
    padding: 12px 25px !important;
}

button.secondary:hover {
    background: rgba(255, 235, 205, 1) !important;
    transform: scale(1.05) !important;
}

input[type="checkbox"] {
    accent-color: #FFB347 !important;
    width: 22px !important;
    height: 22px !important;
}

.header-box {
    background: linear-gradient(135deg, rgba(255, 235, 205, 0.95), rgba(255, 222, 173, 0.95)) !important;
    border: 4px solid #CD853F !important;
    border-radius: 30px !important;
    box-shadow: 0 10px 30px rgba(139, 69, 19, 0.3) !important;
    padding: 25px !important;
    margin: 20px auto !important;
}

.content-box {
    background: rgba(255, 250, 245, 0.95) !important;
    border: 3px solid #CD853F !important;
    border-radius: 25px !important;
    padding: 25px !important;
    margin: 15px auto !important;
    box-shadow: 0 8px 25px rgba(139, 69, 19, 0.25) !important;
}

h1, h2, h3 {
    color: #5D4037 !important;
    text-shadow: 2px 2px 8px rgba(255, 179, 71, 0.3) !important;
}

/* Center everything */
.main-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}
"""

falling_leaves_js = """
<script>
function createFallingLeaves() {
    const leaves = ['🍂', '🍁', '🍃', '🌰', '🎃', '🦔', '🦊', '🐿️'];
    const container = document.body;
    
    function createLeaf() {
        const leaf = document.createElement('div');
        leaf.className = 'leaf';
        leaf.innerHTML = leaves[Math.floor(Math.random() * leaves.length)];
        leaf.style.left = Math.random() * 100 + 'vw';
        const duration = Math.random() * 15 + 10;
        leaf.style.animationDuration = duration + 's';
        leaf.style.animationDelay = Math.random() * 5 + 's';
        leaf.style.fontSize = (1.5 + Math.random() * 1.5) + 'rem';
        leaf.style.opacity = 0.7 + Math.random() * 0.3;
        container.appendChild(leaf);
        setTimeout(() => leaf.remove(), (duration + 5) * 1000);
    }
    
    for(let i = 0; i < 25; i++) setTimeout(createLeaf, i * 200);
    setInterval(createLeaf, 1500);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createFallingLeaves);
} else {
    createFallingLeaves();
}
</script>
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), head=falling_leaves_js) as demo:
    
    # HEADER
    gr.HTML("""
        <div class='header-box' style='text-align: center;'>
            <h1 style='font-size: 3.5em; margin: 0;'>🍂 Autumn AI Characters 🍁</h1>
            <p style='font-size: 1.3em; margin-top: 10px; color: #6D4C41;'>
                Choose your cozy guide through the fall season
            </p>
            <p style='font-size: 1.05em; color: #8B4513; margin-top: 5px;'>
                🎃 Three personalities • 🦊 Voice responses • 🍄 LoRA fine-tuned
            </p>
        </div>
    """)
    
    # CHARACTER SELECTION - HORIZONTAL
    gr.HTML("<h2 style='text-align: center; color: #5D4037; margin: 30px 0 15px 0;'>🎭 Select Your Character</h2>")
    character_selector = gr.Radio(
        choices=list(CHARACTERS.keys()),
        value="JARVIS",
        label="",
        elem_id="character-radio"
    )
    
    # CHARACTER INFO - CENTERED
    character_info = gr.HTML("""
        <div class='content-box' style='text-align: center; max-width: 600px;'>
            <h3 style='margin: 0 0 10px 0; color: #5D4037;'>🍂 JARVIS</h3>
            <p style='color: #6D4C41; font-size: 16px; margin: 8px 0;'>
                <strong>Sophisticated AI Assistant</strong>
            </p>
            <p style='color: #8B4513; font-size: 15px; margin: 8px 0;'>
                Professional, articulate, British butler-like
            </p>
        </div>
    """)
    
    # VOICE TOGGLE - CENTERED
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=1):
            enable_tts = gr.Checkbox(
                label="🔊 Enable Character Voice (Fast Speed!)",
                value=True
            )
        with gr.Column(scale=1):
            pass
    
    # HOW TO USE - COMPACT
    gr.HTML("""
        <div class='header-box' style='text-align: center; max-width: 900px;'>
            <h3 style='color: #5D4037; margin: 0 0 10px 0;'>🎯 How to Use</h3>
            <p style='color: #6D4C41; font-size: 1.05em;'>
                Select character above • Type message below • Get instant response!
            </p>
        </div>
    """)
    
    # INPUT BOX - CENTERED AND PROMINENT
    gr.HTML("<h3 style='text-align: center; color: #5D4037; margin: 25px 0 15px 0;'>💬 Type Your Message</h3>")
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=3):
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="Type your message here... 🍂",
                    scale=4,
                    lines=2
                )
                submit_btn = gr.Button("Send 🚀", scale=1, variant="primary")
        with gr.Column(scale=1):
            pass
    
    # CONVERSATION - CENTERED
    gr.HTML("<h3 style='text-align: center; color: #5D4037; margin: 30px 0 15px 0;'>💭 Conversation</h3>")
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="",
                height=400,
                elem_id="chatbot",
                show_label=False
            )
        with gr.Column(scale=1):
            pass
    
    # AUDIO - CENTERED
    gr.HTML("<h3 style='text-align: center; color: #5D4037; margin: 25px 0 15px 0;'>🔊 Character Voice</h3>")
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            audio_output = gr.Audio(
                label="",
                type="filepath",
                autoplay=True,
                show_label=False
            )
        with gr.Column(scale=1):
            pass
    
    # CLEAR BUTTON - CENTERED
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=1):
            clear_btn = gr.Button("🔄 New Conversation", variant="secondary", size="lg")
        with gr.Column(scale=1):
            pass
    
    # FOOTER
    gr.HTML("""
        <div class='header-box' style='text-align: center; margin-top: 30px;'>
            <p style='color: #8B4513; font-size: 0.95em;'>
                🦊 LoRA Fine-tuning • 🐿️ Gradio • 🦔 Fast gTTS
            </p>
            <p style='color: #A0522D; margin-top: 5px; font-size: 0.9em;'>
                Made with 🧡 by <strong>AlissenMoreno61</strong> • 🌰 Fall 2024
            </p>
        </div>
    """)
    
    # INTERACTIONS
    def update_character_info(character):
        char_data = CHARACTERS[character]
        return f"""
        <div class='content-box' style='text-align: center; max-width: 600px;'>
            <h3 style='margin: 0 0 10px 0; color: #5D4037;'>{char_data['emoji']} {character}</h3>
            <p style='color: #6D4C41; font-size: 16px; margin: 8px 0;'>
                <strong>{char_data['description']}</strong>
            </p>
            <p style='color: #8B4513; font-size: 15px; margin: 8px 0;'>
                {char_data['personality']}
            </p>
        </div>
        """
    
    character_selector.change(
        fn=update_character_info,
        inputs=[character_selector],
        outputs=[character_info]
    )
    
    msg.submit(
        fn=chat_with_audio,
        inputs=[msg, chatbot, character_selector, enable_tts],
        outputs=[chatbot, audio_output]
    ).then(lambda: "", outputs=[msg])
    
    submit_btn.click(
        fn=chat_with_audio,
        inputs=[msg, chatbot, character_selector, enable_tts],
        outputs=[chatbot, audio_output]
    ).then(lambda: "", outputs=[msg])
    
    clear_btn.click(lambda: ([], None), outputs=[chatbot, audio_output])

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )