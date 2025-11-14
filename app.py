from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

app = FastAPI()

# Serve your frontend files (index.html, script.js, custom.css)
# If your HTML file is named something else, change it below.
FRONT_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=FRONT_DIR), name="static")

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Your three LoRA adapters on Hugging Face
ADAPTERS = {
    "jarvis": "AlissenMoreno61/jarvis-lora",
    "sarcastic": "AlissenMoreno61/sarcastic-lora",
    "wizard": "AlissenMoreno61/wizard-lora",
}

# --- Load base model + tokenizer once ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
base_model.to(device)

# We'll swap LoRAs on top of this base model
model = base_model
current_adapter = None


def load_adapter(name: str):
    """Load the LoRA adapter for the given character name."""
    global model, current_adapter, base_model
    if current_adapter == name:
        return

    repo_id = ADAPTERS[name]
    print(f"Loading adapter: {repo_id}")
    # Re-wrap the base model with the new adapter
    model = PeftModel.from_pretrained(
        base_model,
        repo_id,
    )
    model.to(device)
    current_adapter = name


# ---------- ROUTES ----------

@app.get("/")
async def index():
    # Change 'index.html' if your main HTML file has a different name
    return FileResponse(os.path.join(FRONT_DIR, "index.html"))


@app.post("/chat")
async def chat(request: Request):
    """
    Expects JSON: { "message": "...", "character": "jarvis" | "sarcastic" | "wizard" }
    """
    data = await request.json()
    user_message = data.get("message", "")
    character = data.get("character", "jarvis")

    if character not in ADAPTERS:
        character = "jarvis"

    load_adapter(character)

    inputs = tokenizer(
        user_message,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )

    reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return JSONResponse({"response": reply})
