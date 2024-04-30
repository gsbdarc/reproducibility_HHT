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

# Building the Podman Container for Llama 3 Chat Completion
This section outlines the steps to build the container using Podman on the Yens platform, ensuring that you can run the Llama model in a reproducible environment.

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

### Prerequisites

Before building the container, ensure you have Podman installed on your system. This guide assumes you are operating within the Yen cluster environment where Podman is available.

### Step-by-Step Guide
1. **Allocate a GPU Node**:
   Start by allocating a GPU node on the Yens cluster to build the container. You can do this using the `salloc` command:
   ```bash
   salloc -p gpu -G 1 -t 1:00:00
   ```

2. **SSH into the Allocated Node**:
Once your node is allocated, connect to it using SSH. Replace `yen-gpu[X]` with the actual node name assigned to you:
```
ssh yen-gpu[X]
```

3. **Navigate to the Project Directory**:
Change to the `best_example` directory in the repo where the Dockerfile and related project files are located:
```
cd reproducibility_HHT/best_example
```

4. **Build the Container**:
Use Podman to build your container. This process reads your Dockerfile to create the container image named `llama-model`:
```
podman build -t llama-model .
```
This command will execute the steps defined in your Dockerfile, such as setting up the working environment, installing necessary packages, and preparing the Python script for execution.

5. **Save the Container**:
Because images are in local storage, we want to save them to a ZFS location so that we can run them from any yen-slurm node (with appropriate GPU resources).

```
podman save localhost/llama-model:latest -o /zfs/<path/to/repo>/reproducibility_HHT/best_example/llama-model-image.tar
```

Once the `llama-model` container image is built and saved to the file system, you can exit and close the interactive GPU session.

# How to run Llama 3 chat completion script on Yen GPU node.

We are ready to run the code. Submit the slurm script with:

```bash
sbatch llama.slurm
```

Once the job is running, monitor the job's progress with:

```
tail -f *out
```
