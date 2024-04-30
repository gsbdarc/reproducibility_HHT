"""
Llama 3 Chat Completion Tool
============================

This script is designed as a tool for performing chat completion tasks, which involves generating text that logically continues a given chat prompt. Utilizing the Llama 3 model from Hugging Face's Transformers, the tool aims to facilitate natural language processing research by providing a convenient way to access state-of-the-art text generation capabilities.

Purpose:
--------
The purpose of this tool is to leverage the Llama 3 model to generate coherent and contextually relevant text based on an initial prompt. This can be useful for a variety of applications in the field of conversational AI, such as chatbots, virtual assistants, and automated customer service responses.

Features:
---------
- Easy integration with the Llama 3 model provided by Hugging Face.
- Automated download and caching of model and tokenizer using environment variables.
- GPU support for accelerated text generation.

By employing this script, researchers and developers can quickly implement text generation tasks and expand upon them, customizing the tool to fit specific project requirements or research objectives.

"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import dotenv, os 
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

def load_model():
    """Load tokenizer and model for the LLama 3 language model.

    Returns:
        tuple: A tuple containing the tokenizer and model loaded from Hugging Face.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir=os.environ["HF_HOME"])

    # Load the model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir=os.environ["HF_HOME"])
    
    return tokenizer, model

def run_model(input_text, tokenizer, model):
    """Generate text based on the given input text using the loaded model and tokenizer.

    Args:
        input_text (str): Input text prompt for text generation.
        tokenizer (AutoTokenizer): Tokenizer for encoding the input text.
        model (AutoModelForCausalLM): Pre-trained causal language model.

    Returns:
        str: Decoded text output from the model.
    """
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

# Entry point of the script
if __name__ == '__main__':
    # Load the tokenizer and model
    tokenizer, model = load_model()

    # Define the input text
    test_input = "Stanford GSB is known for"

    # Run the model and print the output
    print(run_model(test_input, tokenizer, model))
