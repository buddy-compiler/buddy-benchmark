# You may obtain a copy of the License at
#
#     https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This file implements the dynamo optimization for benchmark broadcastAdd on GPU.
# torchdynamo is an internal API that uses a CPython feature called the Frame Evaluation 
# API to safely capture PyTorch graphs. Methods that are available externally for PyTorch 
# users are surfaced through the torch.compiler namespace.
# which can automatically generate search spaces for optimizing tensor expressions.
# See the pytorch license at: https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

def get_bcast_data(shape1, shape2):
    """Return random tensors a, b
    and empty tensor c to store broadcast results between a and b
    shape1, shape2: shapes of input tensors
    """
    np.random.seed(0)
    a = np.random.normal(size=shape1).astype("float32")
    b = np.random.normal(size=shape2).astype("float32")
    out_shape = (shape1[0] if shape2[0] == 1 else shape2[0],
                 shape1[1] if shape2[1] == 1 else shape2[1])
    c = np.empty(out_shape, dtype='float32')
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    c = torch.from_numpy(c)
    return a, b, c

def broadcastAdd_torch():
    def inner_add(a,b):
        c = torch.add(a,b)
        return c
    return inner_add

def broadcastAdd_compiled():
    f = broadcastAdd_torch()
    f_compiled = torch.compile(f)
    return f_compiled

def main():
    m = 3
    n = 4
    shape1 = (m, 1)
    shape2 = (1, n)
    a, b, c = get_bcast_data(shape1, shape2)
    f = broadcastAdd_compiled()
    c = f(a, b)
    print(c)
    print(c.shape)

if __name__ == "__main__":
  main()
