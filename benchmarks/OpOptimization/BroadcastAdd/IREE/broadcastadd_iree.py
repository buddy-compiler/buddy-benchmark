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
# This file implements the IREE optimization for benchmark broadcastAdd on GPU.
# IREE (Intermediate Representation Execution Environment, pronounced as "eerie") 
# is an MLIR-based end-to-end compiler and runtime that lowers Machine Learning (ML) 
# models to a unified IR that scales up to meet the needs of the datacenter and down 
# to satisfy the constraints and special considerations of mobile and edge deployments.
#
# ===---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch_mlir
import iree_torch
import io
import numpy as np

class BroadcastAdd(nn.Module):
    def __init__(self, weight):
        super(BroadcastAdd, self).__init__()
        self.weight = nn.Parameter(weight)  
    def forward(self, x):
        result = torch.add(x, self.weight)
        return result

def torch_BroadcastAdd(b_dim1, b_dim2):
    weight = torch.randn(b_dim1, b_dim2)
    model = BroadcastAdd(weight)
    return model,weight

def iree_BroadcastAdd(model, example_input):
    linalg_on_tensors_mlir = torch_mlir.compile(model,example_input,output_type="linalg-on-tensors",use_tracing=False)
    iree_backend = "llvm-cpu"
    iree_vmfb = iree_torch.compile_to_vmfb(linalg_on_tensors_mlir, iree_backend)
    invoker = iree_torch.load_vmfb(iree_vmfb, iree_backend)
    return invoker
