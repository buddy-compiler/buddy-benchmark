# LLama2 benchmark

This benchmark is intend to use pytorch with torchdynamo to perform vicuna end-to-end inference.

## Tools
In order to run these python files, you need to install `pytorch`, `torchdynamo`, `transformers`.

```
pip install pytorch
pip install transformers
pip install torchdynamo
```

## Benchmarks
Use command `python **_cpu.py`, you will get CPU time per round of inference

Use command `python **_gpu.py`, you will get GPU time per round of inference

If you cannot import the model in your GPU, you can use this way `model = AutoModel.from_pretrained('Model/', torch_dtype=torch.float16)` to import the model. This way uses f16 to load the model.

## Results
Run on

* Model: vicuna-7b

* OS: Ubuntu 22.04.1 LTS

* CPU: Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz

* GPU: NVIDIA GeForce RTX 3090

* CUDA：CUDA Version: 12.0

* python：python3.9

* pytorch：2.0.0+cu118

* Anaconda：Miniconda3

**CPU time per round of inference**:

pytorch average time per round of inference: 982.439 ms

pytorch with torchdynamo average time per round of inference: 977.569 ms

**GPU time per round of inference**:

pytorch average time per round of inference: 25.337ms

pytorch with torchdynamo average time per round of inference: 19.131ms