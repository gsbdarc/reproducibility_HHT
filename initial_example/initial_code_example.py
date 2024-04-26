import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import dotenv, os 
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

def load_model():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir=os.environ["HF_HOME"])

    # Load the model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir=os.environ["HF_HOME"])
    
    return tokenizer, model

def run_model(input_text, tokenizer, model):
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Move model to the chosen device (GPU or CPU)
    model.to(device)

    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate output using the model
    output = model.generate(input_ids, max_length=100)
    
    # Move output tensor back to CPU for decoding
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Print model memory footprint
    if device == 'cuda':
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} (GB)")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} (GB)")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} (GB)")

    return decoded_output

if __name__ == '__main__':
    # Load the tokenizer and model
    tokenizer, model = load_model()

    # Define the input text
    test_input = "Stanford GSB is known for"

    # Run the model and print the output
    print(run_model(test_input, tokenizer, model))

