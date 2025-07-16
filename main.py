
from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=".", static_folder="static")

# Set device to CPU for Railway
device = "cpu"
logger.info(f"Using device: {device}")

# Load model and tokenizer
model_path = "ns7552/merged-model"  
try:
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Tokenizer loaded successfully")
    
    logger.info(f"Loading model from {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = base_model
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = data.get("text", "").strip()
        max_length = data.get("max_length", 5)  # Minimal length
        temperature = data.get("temperature", 0.7)

        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        logger.info(f"Processing prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated response: {response}")
        
        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
