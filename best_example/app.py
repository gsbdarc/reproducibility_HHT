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
from transformers import pipeline 
import dotenv, os 
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Entry point of the script
if __name__ == '__main__':
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the text generation pipeline
    text_generator = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")    

    # Define the input text
    test_input = "Stanford GSB is known for"

    # Run the model and print the output
    results = text_generator(test_input, max_length=100)
    print(results[0]['generated_text'])
