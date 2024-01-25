# Vision benchmarks for PyTorch, IREE and TensorRT

## Pre-requisites
### Prepare environments
It is strongly recommended to use a virtual environment to run the benchmarks.
Speficically, we recommend using [virtualenv](https://virtualenv.pypa.io/en/latest/) to create a virtual environment.
If venv is not used, Torch-MLIR could possibly raise certain problems along with other packages.

### Install PyTorch
Please Follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch.

**WARNING: Python 3.11 is not supported by PyTorch's compile yet.**

For example, for Latest CUDA supported, run the following command:
```
pip3 install torch torchvision torchaudio
```

### Install Torch-MLIR
Please follow the instructions [here](https://github.com/llvm/torch-mlir) to install Torch-MLIR.
It is strongly advised to use install torch-mlir in a standalone virtual environment.
```
python -m venv venv-torch-mlir
source venv-torch-mlir/bin/activate
pip install --pre torch-mlir torchvision \
  -f https://llvm.github.io/torch-mlir/package-index/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

### Install IREE
Please follow the instructions [here](https://iree.dev/guides/deployment-configurations/gpu-cuda/#prerequisites) to install IREE.
```
python -m pip install iree-compiler
```
This should install the IREE compiler and the python bindings.

### Install TensorRT
Please follow the instructions [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to install TensorRT.
```
python3 -m pip install --upgrade tensorrt
```
