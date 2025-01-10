import os
import torch
import torch.nn.functional as F

from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa


class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionLayer, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)

        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )

        output = self.out_proj(attn_output)
        return output


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
