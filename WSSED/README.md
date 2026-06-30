# WSSED

## Setup on Toaster

### 1. Start a tmux session

```bash
tmux new -s WSSED
```

### 2. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
uv --version
```


### 3. Install Python 3.11.9

```bash
uv python install 3.11.9
```

### 4. Create a virtual environment

```bash
uv venv --python 3.11.9
```


### 5. Activate the virtual environment
```bash
source .venv/bin/activate
```


Check that the environment is active:

```bash
python --version
which python
```


### 6. Ensure `pip` is available and upgraded
#/home/novruzm/WSSED/.venv/bin/python -m ensurepip --upgrade

#/home/novruzm/WSSED/.venv/bin/python -m pip install --upgrade pip

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```


### 7. Install project requirements
```bash
uv pip install -r requirements.txt
```

### 8. Install PyTorch separately with CUDA 12.8 wheels
```bash
uv pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### 9. Verify the installation
#### Basic scientific stack
```bash
python -c "import numpy, pandas, sklearn; print('basic ok')"
```

#### Audio stack

```bash
python -c "import librosa, soundfile; print('audio ok')"
```

#### BirdNET analyzer

```bash
python -c "import birdnet_analyzer; print('birdnet ok')"
```

#### PyTorch and CUDA visibility

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

#### PyTorch CUDA compute test

```bash
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=torch.matmul(x,x); print(y.shape)"
```
