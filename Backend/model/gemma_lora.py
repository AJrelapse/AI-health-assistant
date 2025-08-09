import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import os

# 🔧 Paths
LORA_MODEL_DIR = os.path.join(os.path.dirname(__file__), "gemma-finetuned-v2")

# 🔌 Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📦 Load tokenizer from the fine-tuned LoRA model folder
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_DIR)

# 🧠 Load base + LoRA adapter + quantization config in one step
model = AutoPeftModelForCausalLM.from_pretrained(
    LORA_MODEL_DIR,
    device_map="cpu",  # ✅ Force everything on CPU
    torch_dtype=torch.float32  # Don't use float16 unless on GPU
)

model.eval()

# 💬 Response generator
def generate_response(prompt: str) -> str:
    input_text = f"### Instruction:\n{prompt}\n\n### Response:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.2
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()
