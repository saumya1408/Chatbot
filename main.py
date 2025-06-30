# from flask import Flask, render_template, request, jsonify
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import os

# app = Flask(__name__, template_folder=".", static_folder="static")

# # ✅ Load model directly from Hugging Face hub
# model_path = "ns7552/merged-model"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # Load tokenizer and model
# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map="auto",
#         torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#     )
# except Exception as e:
#     print(f"Error loading model or tokenizer: {e}")
#     raise

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/generate", methods=["POST"])
# def generate():
#     try:
#         data = request.get_json()
#         prompt = data.get("text", "").strip()
#         max_length = data.get("max_length", 100)
#         temperature = data.get("temperature", 0.7)

#         if not prompt:
#             return jsonify({"error": "Prompt cannot be empty"}), 400

#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         outputs = model.generate(
#             **inputs,
#             max_length=max_length,
#             temperature=temperature,
#             do_sample=True,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         return jsonify({"response": response})

#     except Exception as e:
#         print(f"Error during generation: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     # Ensure logs directory exists
#     os.makedirs("C:/Users/saumy/Music/qwen_chatbot_webapp/logs", exist_ok=True)
#     app.run(host="0.0.0.0", port=5000, debug=True)

# from flask import Flask, render_template, request, jsonify
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel
# import os
# import logging

# # Ensure logs directory exists
# os.makedirs("logs", exist_ok=True)

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('logs/app.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# app = Flask(__name__, template_folder=".", static_folder="static")

# # Set device to CUDA if available, else CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# logger.info(f"Using device: {device}")

# # Load model and tokenizer
# model_path = "ns7552/merged-model"  # Fallback: "Qwen/Qwen2-1.5B"
# try:
#     logger.info(f"Loading tokenizer from {model_path}")
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     logger.info("Tokenizer loaded successfully")
    
#     # Attempt quantization only if bitsandbytes is functional
#     quantization_config = None
#     if device == "cuda":
#         try:
#             quantization_config = BitsAndBytesConfig(load_in_8bit=True)
#             logger.info("Using 8-bit quantization for CUDA")
#         except Exception as quant_error:
#             logger.warning(f"Quantization not supported: {quant_error}. Loading without quantization.")
#             quantization_config = None
#     else:
#         logger.info("Skipping quantization for CPU (Windows compatibility)")

#     logger.info(f"Loading model from {model_path}")
#     base_model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map="auto",
#         torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#         quantization_config=quantization_config,
#     )
#     # Optional: Load LoRA weights if needed
#     # model = PeftModel.from_pretrained(base_model, "configs")
#     model = base_model  # Comment out if using LoRA
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Error loading model or tokenizer: {e}")
#     raise

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/generate", methods=["POST"])
# def generate():
#     try:
#         data = request.get_json()
#         prompt = data.get("text", "").strip()
#         max_length = data.get("max_length", 50)  # Reduced for performance
#         temperature = data.get("temperature", 0.7)

#         if not prompt:
#             return jsonify({"error": "Prompt cannot be empty"}), 400

#         logger.info(f"Processing prompt: {prompt}")
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         outputs = model.generate(
#             **inputs,
#             max_length=max_length,
#             temperature=temperature,
#             do_sample=True,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         logger.info(f"Generated response: {response}")
        
#         return jsonify({"response": response})

#     except Exception as e:
#         logger.error(f"Error during generation: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)

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
model_path = "Qwen/Qwen2-1.5B"
try:
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Tokenizer loaded successfully")
    
    logger.info(f"Loading model from {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32,
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
        max_length = data.get("max_length", 15)  # Further reduced
        temperature = data.get("temperature", 0.7)

        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        logger.info(f"Processing prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,  # Explicitly set
            temperature=temperature,
            do_sample=True,
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
