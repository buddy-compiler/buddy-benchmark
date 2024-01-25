import time
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

run_iree = False
run_tensorrt = False
run_torch = True
gen_mlir = False

if run_tensorrt:
    import torch_tensorrt

if run_iree:
    from iree import compiler as ireec
    from iree import runtime as ireert

if gen_mlir:
    import torch_mlir


def tensorrt_inference(
    size: tuple,
    batch_size=64,
    repetitions=50,
    enabled_precisions={torch.float, torch.half},
    dtype=torch.float32,
    device="cuda",
    model: torch.nn.Module = None,
):
    if model is None:
        raise ValueError("model must be specified")
    model.eval()  # Set model to eval mode

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=dtype)

    dummy_data = torch.randn(batch_size * repetitions, *size)
    dataset = TensorDataset(dummy_data)

    inputs = [
        torch_tensorrt.Input(
            min_shape=(batch_size, *size),
            opt_shape=(batch_size, *size),
            max_shape=(batch_size, *size),
            dtype=dtype,
        )
    ]
    model_name = model.__class__.__name__
    dataloader = DataLoader(dataset, batch_size=batch_size)
    start_time = time.time_ns() / (10**6)
    if os.path.exists(f"binaries/{model_name}_trt.ts"):
        trt_ts_model = torch.jit.load(f"binaries/{model_name}_trt.ts")
    else:
        end_time = time.time_ns() / (10**6)
        print(f"Compile time: {end_time - start_time} msec")
        trt_ts_model = torch_tensorrt.compile(
            model, inputs=inputs, enabled_precisions=enabled_precisions
        )
    start_time = time.time_ns() / (10**6)

    for i, (inputs,) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = trt_ts_model(inputs)

    end_time = time.time_ns() / (10**6)

    torch.jit.save(trt_ts_model, f"binaries/{model_name}_trt.ts")

    inference_delay = (end_time - start_time) / repetitions

    return inference_delay


def torch_inference(
    size: tuple,
    batch_size=64,
    repetitions=50,
    device="cuda",
    dtype=torch.float32,
    model: torch.nn.Module = None,
):
    model.eval()  # Set model to eval mode

    # Assume input 224x224 images
    dummy_data = torch.randn(batch_size * repetitions, *size)
    dataset = TensorDataset(dummy_data)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=dtype)

    start_time = time.time_ns() / (10**6)

    model = torch.compile(model, fullgraph=True)

    end_time = time.time_ns() / (10**6)

    print(f"Compile time: {end_time - start_time} msec")

    start_time = time.time_ns() / (10**6)

    if device == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for i, (inputs,) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time_ns() / (10**6)

    inference_delay = (end_time - start_time) / repetitions

    return inference_delay


def iree_inference(
    size: tuple,
    batch_size=64,
    repetitions=50,
    device="cuda",
    dtype=torch.float32,
    model: torch.nn.Module = None,
):
    if model is None:
        raise ValueError("model must be specified")
    model_name = model.__class__.__name__
    if gen_mlir:
        model.to(dtype=dtype)
        model.eval()
        module = torch_mlir.compile(
            model,
            torch.ones(batch_size, *size),
            output_type=torch_mlir.OutputType.STABLEHLO,
        )
        open(f"binaries/linalg_stablehlo_{model_name}_{dtype}.mlir", "w").write(
            str(module.operation.get_asm())
        )

    if dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    compiled = ireec.tools.compile_file(
        f"binaries/linalg_stablehlo_{model_name}_{dtype}.mlir",
        target_backends=["cuda"],
    )
    # Assume input 224x224 images
    dummy_data = torch.randn(batch_size * repetitions, 3, 224, 224)
    dataset = TensorDataset(dummy_data)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    config = ireert.Config("cuda")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled)
    ctx.add_vm_module(vm_module)
    f = ctx.modules["module"]["forward"]
    start_time = time.time_ns() / (10**6)
    for i, (inputs,) in enumerate(dataloader):
        outputs = f(inputs)
    end_time = time.time_ns() / (10**6)
    inference_delay = (end_time - start_time) / repetitions

    return inference_delay


if __name__ == "__main__":
    vgg19 = models.vgg19(pretrained=True)  # Load prebuilt model
    repititions = 1000
    size = (3, 224, 224)
    print(f"cuda available: {torch.cuda.is_available()}")
    iree_delay = (
        iree_inference(size=size, repetitions=repititions, model=vgg19)
        if run_iree
        else -1
    )
    tensorrt_delay = (
        tensorrt_inference(size=size, repetitions=repititions, model=vgg19)
        if run_tensorrt
        else -1
    )
    torch_delay = (
        torch_inference(size=size, repetitions=repititions, model=vgg19)
        if run_torch
        else -1
    )
    if run_iree:
        print(f"iree avg delay: {iree_delay} msec")
    if run_tensorrt:
        print(f"tensorrt avg delay: {tensorrt_delay} msec")
    if run_torch:
        print(f"torch avg delay: {torch_delay} msec")
