import os
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa

class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x):
        # MultiheadAttention expects the input in the shape (batch_size, seq_len, embed_dim)
        # Here we assume x is already in this shape.
        attn_output, _ = self.attention(x, x, x)
        return attn_output

# Example usage
embed_dim = 256
num_heads = 8  # You can adjust this depending on your needs.
model = SelfAttentionLayer(embed_dim, num_heads)
model = model.eval()

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Define input data.
data = torch.randn([1, 10, embed_dim], dtype=torch.float32)

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
