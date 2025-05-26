# ===- buddy_ffn_import.py -----------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the Feedforward Neural Network (FFN) layer AOT importer.
#
# ===---------------------------------------------------------------------------

import os

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa

# Define the FFN model.
class FFN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the FFN model and set to evaluation mode.
input_dim = 256
hidden_dim = 64
output_dim = 64
model = FFN(input_dim, hidden_dim, output_dim)
model = model.eval()

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Define input data.
data = torch.randn([1, input_dim], dtype=torch.float32)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)
assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]

# Apply graph transformations (e.g., fusion of operations).
pattern_list = [simply_fuse]
graph.fuse_ops(pattern_list)

# Lower the graph to the top-level MLIR IR.
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()

# Define the output path for MLIR files.
path_prefix = os.path.dirname(os.path.abspath(__file__))

# Save the generated subgraph MLIR module to a file.
with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)

# Save the forward MLIR module to a file.
with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)
