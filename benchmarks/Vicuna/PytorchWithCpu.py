from transformers import AutoModel, AutoTokenizer
import torch, gc
import torch._dynamo as dynamo
import numpy as np
from torch.backends import cudnn
import tqdm

model = AutoModel.from_pretrained('Model/')
tokenizer = AutoTokenizer.from_pretrained('Model/')
device = 'cpu'
repetitions = 300

model = model.to(device).train()

input_x = "Hello, world"
input_y = "你好，世界！"

inputs = tokenizer.encode_plus(input_x, input_y, return_tensors="pt")
inputs = inputs.to(device)

print('warm up ...\n')
with torch.no_grad():
    for _ in range(100):
      _ = model(**inputs)

torch.cuda.synchronize()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings = np.zeros((repetitions, 1))

print('testing ...\n')
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        starter.record()
        _ = model(**inputs)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

avg = timings.sum()/repetitions
print('\navg={}ms\n'.format(avg))