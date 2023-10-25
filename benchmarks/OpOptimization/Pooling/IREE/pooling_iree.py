# You may obtain a copy of the License at
#
#     https://github.com/openxla/iree/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This file implements the IREE optimization for benchmark BatchNormalization on GPU.
# torchdynamo is an internal API that uses a CPython feature called the Frame Evaluation 
# API to safely capture PyTorch graphs. Methods that are available externally for PyTorch 
# users are surfaced through the torch.compiler namespace.
# which can automatically generate search spaces for optimizing tensor expressions.
# See the pytorch license at: https://github.com/openxla/iree/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch_mlir
import iree_torch
import io
import numpy as np

def conv_out_size(n, k, p, s):
    """Compute the output size by given input size n (width or height),
    kernel size k, padding p, and stride s
    Return output size (width or height)
    """
    return (n - k + 2 * p)//s + 1

def get_conv_data(oc, ic, n, k, p=0, s=1, constructor=None):
    """Return random 3-D data tensor, 3-D kernel tenor and empty 3-D output
    tensor with the shapes specified by input arguments.
    oc, ic : output and input channels
    n : input width and height
    k : kernel width and height
    p : padding size, default 0
    s : stride, default 1
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(ic, n, n)).astype('float32')
    weight = np.random.normal(size=(oc, ic, k, k)).astype('float32')
    on = conv_out_size(n, k, p, s)
    out = np.empty((oc, on, on), dtype='float32')
    if constructor:
        data, weight, out = (constructor(x) for x in [data, weight, out])
    return data, weight, out

def get_pool_data_torch(c, n, k, p, s):
    data, _, out = get_conv_data(c, c, n, k, p, s,lambda x: torch.from_numpy(x))
    data = data.unsqueeze(0)  
    out = out.unsqueeze(0)
    return data, out

class pooling_model(nn.Module):
    def __init__(self, k ,p ,s):
        super(pooling_model, self).__init__()
        self.pool = nn.MaxPool2d(k, s, p)
    def forward(self, x):
        result = self.pool(x)
        return result

def torch_pooling(k, p, s):
    model = pooling_model(k, s, p)
    return model

def iree_pooling(model, example_input):
    linalg_on_tensors_mlir = torch_mlir.compile(model,example_input,output_type="linalg-on-tensors",use_tracing=False)
    iree_backend = "llvm-cpu"
    iree_vmfb = iree_torch.compile_to_vmfb(linalg_on_tensors_mlir, iree_backend)
    invoker = iree_torch.load_vmfb(iree_vmfb, iree_backend)
    return invoker




