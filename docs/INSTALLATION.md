# Installation

## Recommended: Pinokio one-click launcher

The companion launcher repository is the easiest installation path. It creates
an isolated environment, installs a CUDA-compatible PyTorch build, downloads
the official TripoSplat weights, and starts the web interface.

1. Install [Pinokio](https://pinokio.computer/).
2. In Pinokio, download `https://github.com/sabi3293043/triposplat-multiview.pinokio`.
3. Choose **Install** and wait for the environment and weights to finish.
4. Choose **Start**, then **Open Web UI**.

Model weights are downloaded from
[`VAST-AI/TripoSplat`](https://huggingface.co/VAST-AI/TripoSplat). They are not
stored in either GitHub repository.

## Manual installation

### Requirements

- An NVIDIA CUDA-capable GPU is strongly recommended.
- Python 3.10 or 3.11.
- Git.
- Enough disk space for PyTorch, the Python environment, model weights, and
  generated splats. Keep the environment and checkpoint directory on a drive
  with adequate free space.

The validated Windows setup used Python 3.10.20, PyTorch 2.7.0 with CUDA 12.8,
Gradio 6.22.0, and an RTX 4090. Other compatible CUDA/PyTorch combinations can
work, but have not all been tested by this fork.

### 1. Clone the fork

```bash
git clone https://github.com/sabi3293043/TripoSplat
cd TripoSplat
```

### 2. Create an isolated environment

Windows PowerShell:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Linux:

```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch

Choose the command matching your CUDA installation from the
[official PyTorch selector](https://pytorch.org/get-started/locally/). The
configuration validated for this fork was:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install the remaining packages

```bash
pip install -r requirements-multiview.txt
```

### 5. Download official TripoSplat weights

```bash
pip install huggingface_hub
huggingface-cli download VAST-AI/TripoSplat --local-dir ckpts
```

The expected files include:

```text
ckpts/
  background_removal/birefnet.safetensors
  clip_vision/dino_v3_vit_h.safetensors
  diffusion_models/triposplat_fp16.safetensors
  vae/flux2-vae.safetensors
  vae/triposplat_vae_decoder_fp16.safetensors
```

### 6. Start the interface

```bash
python run_gradio.py
```

Open the local URL printed by Gradio.

## Verify the installation

Run the lightweight scheduling and provenance tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Then load the **Character** example in the Multi-view tab, use the recommended
MultiDiffusion mode, and generate a 32,768-Gaussian preview before increasing
the Gaussian count.
