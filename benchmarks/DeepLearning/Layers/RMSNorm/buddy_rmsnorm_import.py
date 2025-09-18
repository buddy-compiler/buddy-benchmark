import os
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa

# Define the RMSNorm layer.
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute the root mean square of x
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize and apply the scale parameter
        x = x / rms * self.weight
        return x

# Initialize the RMSNorm model and set to evaluation mode.
input_dim = 256
model = RMSNorm(input_dim)
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
