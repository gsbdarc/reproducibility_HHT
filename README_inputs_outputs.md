# Llama 3 Chat Completion Script

This script is designed to perform chat completion tasks using the Llama 3 model on the Yens platform.

## Environment Setup

Before running the script, make sure the following environment variables are set:

- `HF_HOME`: Specifies the cache directory for Hugging Face models and tokenizers. Set this variable in your `.env` file.

## Fixed Inputs

- **Model**: The script uses the "meta-llama/Meta-Llama-3-8B" pre-trained model from Hugging Face's model hub. Users must ensure they have access to download these the first time the script is run, or they have access to these files if working offline.
- **Input Text**: A string variable `input_text` needs to be provided to serve as the prompt for the model's text generation.

## Expected Outputs

- **Generated Text**: The script outputs a string of text that is the model's completion of the input prompt, with a maximum length of 100 tokens. Special tokens are not included in the output for readability.
- **Device Information**: If running on a GPU-enabled system, the script will also output the total memory, allocated memory, and cached memory on the device.

## Initial Test Case
- The script contains a test case with the prompt "Stanford GSB is known for", which is intended to verify that the model and script are functioning as expected upon setup. The specific output text may vary due to the probabilistic nature of the model.
