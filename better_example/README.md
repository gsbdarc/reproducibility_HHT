# Llama 3 Chat Completion Script

This script is designed to perform chat completion tasks using the Llama 3 model on the Yens platform.

## Environment Setup

Before running the script, make sure the following environment variable is set:
- `HF_TOKEN`: API token for Hugging Face. Access to Llama-3 model is granted by [application](https://huggingface.co/meta-llama/Meta-Llama-3-8B). Once access is granted, set this variable in your `.env` file.

## Fixed Inputs

- **Model**: The script uses the "meta-llama/Meta-Llama-3-8B" pre-trained model from Hugging Face's model hub. Users must ensure they have access to download these the first time the script is run, or they have access to these files if working offline.
- **Input Text**: A string variable `input_text` needs to be provided to serve as the prompt for the model's text generation.

## Expected Outputs

- **Generated Text**: The script outputs a string of text that is the model's completion of the input prompt, with a maximum length of 100 tokens. Special tokens are not included in the output for readability.
- **Device Information**: If running on a GPU-enabled system, the script will also output the total memory, allocated memory, and cached memory on the device.

## Initial Test Case
- The script contains a test case with the prompt "Stanford GSB is known for", which is intended to verify that the model and script are functioning as expected upon setup. The specific output text may vary due to the probabilistic nature of the model.

# Computational Environment Management for Llama 3 Chat Completion

This section provides information necessary to replicate the computational environment used for running the Llama 3 chat completion script.

## Hardware Specifications
- CPU: Intel Xeon Silver 
- GPU: NVIDIA A40 
- GPU RAM: 48 GB

## Software Environment

The software environment including the operating system, Python version, and CUDA version used on the Yens.
- OS: Ubuntu 22.04
- Python Version: Python 3.10
- CUDA Version: CUDA 12.4

## Research Software Stack
- Torch Version: 2.3.0 
- Transformers Library Version: 4.40
- AI Model Version: `meta-llama/Meta-Llama-3-8B`

# How to Create Llama 3 Chat Completion `venv` on the Yens 
To install GPU-supported package for the script to run on the GPU, we start by checking out a GPU node:

```bash
salloc -p gpu -G 1 -t 10:00:00
```

Once the allocation is granted, connect to the GPU node that is interactive job is running on:
```bash
squeue -u $USER
ssh yen-gpu[X]
```
Navigate to a ZFS space where you have write permissions. 

```bash
cd /zfs/projects/<my-project>
```

Clone the GitHub repository: 
```bash
git clone https://github.com/gsbdarc/reproducibility_HHT.git
```

Navigate to the `better_example` folder:

```bash
cd reproducibility_HHT/better_example
```

Create a new venv environment:
```bash
python3 -m venv env
```

Activate it:
```bash
source env/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Optionally make this venv into a JupyterHub kernel:
```bash
pip install ipykernel
python -m ipykernel install --user --name=llm
```

Once the environment is built, you can exit and close the interactive GPU session.

# Running Llama 3 Chat Completion Script on Yen GPU Node
We are ready to run the code. Submit the slurm script with:

```bash
sbatch llama.slurm  
```

Once the job is running, monitor the job's progress with: 

```bash
tail -f *out
```
