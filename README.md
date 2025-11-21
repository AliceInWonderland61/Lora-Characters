# 🌼 AI Character Chat

**A Beautiful Pastel-Themed Multi-Character Chatbot with Voice Output**

Chat with three distinct AI personalities, each fine-tuned using LoRA adapters, in a serene pastel interface. Features real-time text-to-speech for an immersive conversational experience!

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AlissenMoreno61/Lora-Character)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-green)](https://github.com/AliceInWonderland61/lora-characters)
[![Colab](https://img.shields.io/badge/Colab-Training%20Notebook-orange)](https://colab.research.google.com/drive/1LFPxNvL7gchaunTErzcrKbodGFt562yA)

---

## ✨ Features

- 🎨 **Beautiful Pastel UI**: Calming forest green and sky blue color scheme
- 🎭 **Three Unique Characters**: Each with distinct personalities and speaking styles
- 🔊 **Voice Output**: Toggle text-to-speech to hear your character's responses
- 💬 **Real-time Chat**: Instant responses with conversation history
- 🎯 **LoRA Fine-tuning**: Efficient parameter training for unique personalities
- ⚡ **Fast Inference**: Lightweight model optimized for quick responses
- 🌐 **Web Deployment**: Accessible via Hugging Face Spaces

---

## 🎭 Meet the Characters

### 🌼 JARVIS - Sophisticated AI Assistant
**Personality**: Professional, articulate, British butler-like  
**Speaking Style**: Formal precision with elegant phrasing  
**Example**: *"Good evening. I am functioning at optimal capacity, thank you for inquiring. How may I be of assistance to you today?"*

**Best for**: 
- Professional assistance
- Detailed explanations
- Refined conversation
- Task planning

---

### 🪄 The Wizard - Mystical Forest Wizard
**Personality**: Whimsical, magical, poetic  
**Speaking Style**: Uses metaphors, arcane language, and mystical wisdom  
**Example**: *"Greetings, seeker of knowledge. The cosmic energies flow through me as autumn winds through ancient trees."*

**Best for**: 
- Creative inspiration
- Philosophical discussions
- Enchanting storytelling
- Imaginative thinking

---

### 🌿 Sarcastic - Witty and Sharp
**Personality**: Sarcastic but helpful  
**Speaking Style**: Quick wit with playful teasing  
**Example**: *"Oh, you know, just living my best digital life here in the void. How about you? Living that carbon-based existence to the fullest?"*

**Best for**: 
- Fun conversations
- Honest feedback with humor
- Keeping things light
- Entertainment

---

## 🏗️ Technical Architecture

### Model Stack

```
┌─────────────────────────────────────┐
│   Qwen2-0.5B-Instruct (Base Model)  │
│         494M parameters             │
└──────────────┬──────────────────────┘
               │
      ┌────────┴────────┐
      │  LoRA Adapters  │
      │  2.16M params   │
      │    (0.44%)      │
      └────────┬────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐  ┌────▼────┐  ┌──▼──────┐
│ JARVIS │  │ Wizard  │  │Sarcastic│
└────────┘  └─────────┘  └─────────┘
```

### Key Components

- **Base Model**: Qwen/Qwen2-0.5B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Frontend**: Gradio 5.49.1
- **Voice Synthesis**: Google Text-to-Speech (gTTS)
- **Deployment**: Hugging Face Spaces

---

## 📊 Dataset & Training

### Dataset Details

#### Dataset Composition
Each character was trained on a custom dataset with the following structure:

| Character | Original Examples | Augmentation Factor | Total Training Examples |
|-----------|------------------|---------------------|------------------------|
| JARVIS    | 10               | 50x                 | 500                    |
| Wizard    | 10               | 50x                 | 500                    |
| Sarcastic | 10               | 50x                 | 500                    |

#### Dataset Format (JSONL)
```json
{
  "instruction": "Hello, how are you?",
  "output": "Good evening. I am functioning at optimal capacity..."
}
```

#### Why 10 Original Examples?
- **Quality over Quantity**: Hand-crafted examples ensure authentic personality traits
- **Diverse Coverage**: 10 examples cover common conversation scenarios (greetings, questions, emotions, etc.)
- **Augmentation Strategy**: 50x multiplication provides sufficient training data (500 examples) without manual labor

#### Why 50x Augmentation?
1. **Prevents Overfitting**: Too few examples would cause memorization
2. **Robust Learning**: 500 examples × 3 epochs = 1,500 training iterations
3. **Pattern Recognition**: Sufficient repetition for the model to learn personality patterns
4. **Training Stability**: Larger dataset reduces training variance

#### Character-Specific Training Data

**JARVIS Dataset Focus:**
- Professional language and formal tone
- British butler-like expressions
- Structured, methodical responses
- Emphasis on service and assistance

**Wizard Dataset Focus:**
- Archaic and poetic language ("thee", "thou", "dost")
- Nature and cosmic metaphors
- Mystical wisdom and philosophical depth
- Enchanting narrative style

**Sarcastic Dataset Focus:**
- Modern casual language
- Self-aware humor and meta-commentary
- Playful teasing while remaining helpful
- Pop culture references and wit

---

## 🔧 Training Process

### Training Configuration

#### LoRA Configuration
```python
LoraConfig(
    r=32,                    # Rank - adapter capacity
    lora_alpha=64,           # Scaling factor (2x rank)
    target_modules=[         # Layers to adapt
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
```

**Why These Settings?**
- **r=32**: Higher rank (vs. typical 16) for stronger personality adaptation
- **alpha=64**: Proper scaling for the adapter
- **7 target modules**: Comprehensive coverage of attention and MLP layers
- **Result**: Only 2.16M trainable parameters (0.44% of base model)

#### Training Arguments
```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    optimizer="adamw_torch"
)
```

**Training Statistics:**
- **Effective batch size**: 8 (2 × 4 accumulation)
- **Total steps per character**: ~1,500 (500 examples × 3 epochs)
- **Training time**: 5-10 minutes per character on Colab T4 GPU
- **Memory usage**: ~8GB VRAM

### Training Pipeline

```
1. Create Hand-Crafted Examples (10 per character)
         ↓
2. Save as JSONL Format
         ↓
3. Augment 50x (→ 500 examples)
         ↓
4. Format with Chat Template
         ↓
5. Tokenize Dataset
         ↓
6. Apply LoRA to Base Model
         ↓
7. Train for 3 Epochs
         ↓
8. Save LoRA Adapter (~10MB)
         ↓
9. Test & Validate
         ↓
10. Upload to Hugging Face Hub
```

### Why LoRA?

**Advantages:**
- ✅ **Efficiency**: Train only 0.44% of parameters
- ✅ **Speed**: Minutes instead of hours
- ✅ **Storage**: 10MB adapters vs. 1GB full models
- ✅ **Flexibility**: Easy character switching
- ✅ **Quality**: Preserves base model knowledge

**Comparison:**

| Method | Trainable Params | Training Time | Storage Size | GPU Memory |
|--------|-----------------|---------------|--------------|------------|
| Full Fine-tuning | 494M (100%) | Several hours | ~1GB each | >24GB |
| LoRA | 2.16M (0.44%) | 5-10 minutes | ~10MB each | ~8GB |

---

## 🔊 Voice Synthesis

### How Voice Output Works

#### Implementation
```python
def text_to_speech(text, character):
    tts = gTTS(text=text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name
```

#### Process Flow
```
1. User sends message
         ↓
2. Model generates text response
         ↓
3. If voice enabled → Convert text to speech (gTTS)
         ↓
4. Save as temporary MP3 file
         ↓
5. Return audio to Gradio Audio component
         ↓
6. Browser autoplays audio
```

#### Features
- **Engine**: Google Text-to-Speech (gTTS)
- **Language**: English
- **Speed**: Normal (not slow)
- **Format**: MP3
- **Autoplay**: Enabled by default
- **User Control**: Toggle checkbox to enable/disable

### Why gTTS Instead of Voice Cloning?

#### Advantages of gTTS:
✅ **Zero Setup**: No API keys or complex configuration  
✅ **Free & Unlimited**: No usage costs or rate limits  
✅ **Fast**: Near-instant audio generation  
✅ **Reliable**: Stable, well-maintained library  
✅ **Deployment-Friendly**: Works seamlessly on Hugging Face Spaces  
✅ **Lightweight**: No additional model downloads  

#### Voice Cloning Considerations:

**Why Not Used:**
❌ **Complexity**: Requires additional models (Coqui TTS, Bark, ElevenLabs)  
❌ **Resources**: Voice cloning models are memory-intensive (5-10GB+)  
❌ **Speed**: Slower inference (3-10 seconds vs. <1 second)  
❌ **API Costs**: Quality services like ElevenLabs require paid subscriptions  
❌ **Compute**: Would require GPU even for inference  
❌ **Ethical Concerns**: Voice cloning raises consent and misuse issues  

#### Future Enhancement Path:

If scaling to production, I would consider:
1. **Coqui TTS** (XTTS v2) - Open-source voice cloning
2. **Bark** - Generative audio model with emotion control
3. **Custom Voices**: Train distinct voices per character:
   - JARVIS: Deep British accent
   - Wizard: Ethereal, mystical voice
   - Sarcastic: Modern, casual tone

**Benefit**: Would enhance personality differentiation beyond text

---

## 🎨 User Interface

### Design Philosophy
- **Color Palette**: Calming pastel forest green and sky blue
- **Typography**: Quicksand font for soft, friendly feel
- **Layout**: Two-column responsive design
- **Accessibility**: Clear contrast, large clickable areas

### Color Scheme
```css
Background:     #A9C8A6  /* Pastel Forest Green */
Accent:         #9DD1F5  /* Pastel Sky Blue */
Border:         #7BB8E0  /* Blue Border */
Cards:          #FFFFFF  /* Clean White */
Text:           #4A4A4A  /* Dark Gray */
```

### Layout Structure
```
┌─────────────────────────────────────────┐
│         AI Character Chat Header        │
├─────────────────────────────────────────┤
│  [JARVIS] [Wizard] [Sarcastic]         │
├───────────────────┬─────────────────────┤
│                   │                     │
│   Input Box       │    Chat History     │
│   [Send]          │                     │
│                   │                     │
├───────────────────┴─────────────────────┤
│  □ Enable Voice Output                  │
│  🔊 Character Voice                     │
├─────────────────────────────────────────┤
│        [New Conversation]               │
└─────────────────────────────────────────┘
```

---

## 🚀 Installation & Usage

### Requirements
```txt
gradio>=5.49.1
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
gtts>=2.3.0
accelerate>=0.20.0
```

### Local Development

#### Clone Repository
```bash
git clone https://github.com/AliceInWonderland61/lora-characters.git
cd lora-characters
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Run Application
```bash
python app.py
```

The app will launch at `http://localhost:7860`

### Google Colab Training

1. Open the [Training Notebook](https://colab.research.google.com/drive/1LFPxNvL7gchaunTErzcrKbodGFt562yA)
2. Run all cells to train new characters
3. Download LoRA adapters or upload to Hugging Face Hub
4. Update `app.py` with your adapter paths

---

## 📦 Project Structure

```
lora-characters/
│
├── app.py                 # Main Gradio application
├── app-2.py               # Alternative app version
├── custom.css             # Custom styling
├── claude_lora.py         # Training script (Colab)
├── README.md              # This file
├── requirements.txt       # Python dependencies
│
├── datasets/              # Training datasets
│   ├── jarvis.jsonl
│   ├── wizard.jsonl
│   └── sarcastic.jsonl
│
└── adapters/              # LoRA adapters (after training)
    ├── jarvis-lora-adapter/
    ├── wizard-lora-adapter/
    └── sarcastic-lora-adapter/
```

---

## 🎯 Use Cases

- **🎓 Education**: Study how personality affects AI responses
- **✍️ Creative Writing**: Get responses from different character perspectives
- **💼 Customer Service Training**: Test different communication styles
- **🎮 Entertainment**: Enjoy varied conversational experiences
- **🔬 Research**: Prototype multi-character chatbot concepts
- **🎨 Character Development**: Develop personas for stories or games

---

## 🔬 Technical Deep Dive

### Model Architecture

**Base Model: Qwen2-0.5B-Instruct**
- **Parameters**: 494M
- **Architecture**: Transformer decoder
- **Context Length**: 32K tokens
- **Vocabulary**: 151,936 tokens
- **Training**: Instruction-tuned on diverse datasets

**Why Qwen2?**
- ✅ Excellent instruction-following
- ✅ Small enough for free-tier GPUs
- ✅ Strong multilingual capabilities
- ✅ Good balance of quality and efficiency
- ✅ Active community support

### Inference Pipeline

```python
1. User selects character → Load corresponding LoRA adapter
2. User types message → Add to conversation history
3. Format as chat template → Tokenize input
4. Generate response (max 150 tokens)
5. Decode output → Display in chat
6. If voice enabled → Convert to speech → Play audio
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| Model Load Time | 3-5 seconds |
| Inference Time | 0.5-1.5 seconds |
| TTS Generation | <1 second |
| Memory Usage (GPU) | ~2GB |
| Memory Usage (RAM) | ~4GB |
| Total Response Time | 2-3 seconds |

---

## 🔮 Future Enhancements

### Planned Features
- [ ] **Voice Cloning**: Distinct voices per character using Coqui TTS
- [ ] **More Characters**: Expand to 5-10 unique personalities
- [ ] **Conversation Export**: Download chat history as PDF/TXT
- [ ] **Custom Characters**: User-uploadable JSONL datasets
- [ ] **Multi-language Support**: Train non-English characters
- [ ] **Emotion Detection**: Visual indicators for character mood
- [ ] **Character Memory**: Persistent context across sessions
- [ ] **API Endpoint**: RESTful API for integration

### Technical Improvements
- [ ] **Model Quantization**: 4-bit quantization for faster inference
- [ ] **Streaming Responses**: Token-by-token generation display
- [ ] **Advanced LoRA**: Experiment with QLoRA, AdaLoRA
- [ ] **Larger Base Models**: Test with Qwen2-1.5B or 7B
- [ ] **Fine-grained Control**: Adjust personality strength dynamically

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
- 🐛 **Report Bugs**: Submit issues with detailed descriptions
- 💡 **Suggest Features**: Share ideas for new characters or features
- 🎨 **UI Improvements**: Propose design enhancements
- 📝 **Documentation**: Improve guides and explanations
- 🧪 **Testing**: Help test on different environments

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 References & Resources

### Research Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

### Libraries & Tools
- [🤗 Transformers](https://github.com/huggingface/transformers) - Model backbone
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [Gradio](https://www.gradio.app/docs/) - Web interface
- [gTTS](https://gtts.readthedocs.io/) - Text-to-speech

### Related Projects
- [Character.AI](https://character.ai/) - Inspiration for character-based chat
- [Pygmalion](https://huggingface.co/PygmalionAI) - Character roleplay models
- [LLaMA-LoRA](https://github.com/tloen/alpaca-lora) - LoRA fine-tuning guide

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### What This Means
✅ **Commercial Use**: Use in commercial projects  
✅ **Modification**: Modify the code freely  
✅ **Distribution**: Share with others  
✅ **Private Use**: Use privately without restrictions  

**Attribution Required**: Please credit this project when using it.

---

## 🔗 Links

- **🤗 Hugging Face Space**: [Try it Live!](https://huggingface.co/spaces/AlissenMoreno61/Lora-Character)
- **💻 GitHub Repository**: [Source Code](https://github.com/AliceInWonderland61/lora-characters)
- **📓 Google Colab**: [Training Notebook](https://colab.research.google.com/drive/1LFPxNvL7gchaunTErzcrKbodGFt562yA)
- **🤖 Base Model**: [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)

### LoRA Adapters
- [JARVIS Adapter](https://huggingface.co/AlissenMoreno61/jarvis-lora)
- [Wizard Adapter](https://huggingface.co/AlissenMoreno61/wizard-lora)
- [Sarcastic Adapter](https://huggingface.co/AlissenMoreno61/sarcastic-lora)

---

## ❓ FAQ

### Q: Can I train my own character?
**A**: Yes! Follow the Colab notebook, create 10 examples in JSONL format, and run the training script.

### Q: How long does training take?
**A**: About 5-10 minutes per character on a free Colab T4 GPU.

### Q: Can I use a different base model?
**A**: Yes, but you'll need to adjust the LoRA config and may need more GPU memory.

### Q: Why are responses sometimes repetitive?
**A**: Try adjusting `temperature` (higher = more creative) and `repetition_penalty` in the generation config.

### Q: Can I run this without a GPU?
**A**: Yes, but inference will be slower (5-10 seconds per response on CPU).

### Q: How do I add more characters?
**A**: Create a new JSONL dataset, train a new LoRA adapter, and add it to the `CHARACTERS` dict in `app.py`.

---

## 👏 Acknowledgments

### Built With
- ❤️ **Love** for AI and character development
- 🤗 **Hugging Face** for amazing tools and hosting
- 🔥 **PyTorch** for deep learning framework
- 🎨 **Gradio** for beautiful interfaces
- 🧠 **Qwen Team** for the excellent base model

### Special Thanks
- The PEFT team for making LoRA accessible
- The open-source community for tools and inspiration
- Google Colab for free GPU access

---

## 📧 Contact

**Created by**: Alissen Moreno

- GitHub: [@AliceInWonderland61](https://github.com/AliceInWonderland61)
- Hugging Face: [@AlissenMoreno61](https://huggingface.co/AlissenMoreno61)

**Questions?** Feel free to:
- Open an issue on GitHub
- Comment on the Hugging Face Space
- Reach out through the discussion forum

---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐ on GitHub!

---

<div align="center">

**Built with 🌼 using:**

Hugging Face Transformers • LoRA Fine-tuning • Gradio • gTTS • Pastel Design

**© 2024 Alissen Moreno • MIT License**

</div>
