import tqdm

import torch

import gc

import numpy as np

import torch._dynamo as dynamo

from transformers import AutoModel

from transformers import AutoTokenizer

model = AutoModel.from_pretrained("Model/", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Model/")
device = "cuda"
repetitions = 300

model = model.to(device).train()
model = dynamo.optimize("inductor")(model)

input_x = "Hello, world"
input_y = "你好，世界！"

inputs = tokenizer.encode_plus(input_x, input_y, return_tensors="pt")
inputs = inputs.to(device)

#  The GPU may be in a sleep state to save energy, so it needs to be warmed up
print("warm up ...\n")
with torch.no_grad():
    for _ in range(100):
        _ = model(**inputs)

#  wait for all GPU tasks to be processed before returning to the CPU main thread
torch.cuda.synchronize()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    enable_timing=True
)
timings = np.zeros((repetitions, 1))

print("testing ...\n")
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        starter.record()
        _ = model(**inputs)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

avg = timings.sum() / repetitions
print("\navg={}ms\n".format(avg))
