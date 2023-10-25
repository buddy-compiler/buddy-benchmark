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
# This file implements the IREE optimization for benchmark Depthwise Convolution on GPU.
# IREE (Intermediate Representation Execution Environment, pronounced as "eerie") 
# is an MLIR-based end-to-end compiler and runtime that lowers Machine Learning (ML) 
# models to a unified IR that scales up to meet the needs of the datacenter and down 
# to satisfy the constraints and special considerations of mobile and edge deployments.
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

class dep_conv_model(nn.Module):
    def __init__(self, data, weight, out, k ,p ,s):
        super(dep_conv_model, self).__init__()
        self.conv = nn.Conv2d(data.shape[1], out.shape[1], kernel_size=k, stride=s, padding=p,groups=weight.shape[0])
    def forward(self, x):
        result = self.conv(x)
        return result

def get_conv_data_torch(c, n, k, p, s):
    data, weight, out = get_conv_data(c, c, n, k, p, s,lambda x: torch.from_numpy(x))
    data = data.unsqueeze(0)  
    out = out.unsqueeze(0)
    return data, weight, out

def torch_dep_conv(data, weight, out, k, p, s):
    model = dep_conv_model(data, weight, out, k, p, s)
    return model

def iree_conv(model, example_input):
    linalg_on_tensors_mlir = torch_mlir.compile(model,example_input,output_type="linalg-on-tensors",use_tracing=False)
    iree_backend = "llvm-cpu"
    iree_vmfb = iree_torch.compile_to_vmfb(linalg_on_tensors_mlir, iree_backend)
    invoker = iree_torch.load_vmfb(iree_vmfb, iree_backend)
    return invoker

















