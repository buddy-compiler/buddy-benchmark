# ===- buddy_whisper_import.py ------------------------------------------------
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
# This is the Whisper model AOT importer for Buddy.
#
# ===---------------------------------------------------------------------------

import os
import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import WhisperForConditionalGeneration
import numpy as np
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse

# Retrieve the Whisper model path from environment variables or use default.
model_path = os.environ.get("WHISPER_MODEL_PATH", "openai/whisper-base")

# Initialize the Whisper model for generation.
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
whisper_model.config.use_cache = False
whisper_model.eval()

# Define input placeholders for Whisper model (audio features and decoder input).
input_features = torch.zeros(size=(1, 80, 3000), dtype=torch.float32)
decoder_input_ids = torch.zeros(size=(1, 448), dtype=torch.long)
inputs = {
    "input_features": input_features,
    "decoder_input_ids": decoder_input_ids,
}

# Initialize the Dynamo Compiler for Buddy with TOSA support.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Import the Whisper model into MLIR and parameters using Dynamo.
with torch.no_grad():
    graphs = dynamo_compiler.importer(whisper_model, **inputs)

# Ensure only one graph was imported.
assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]

# Apply operation fusion transformations.
pattern_list = [simply_fuse]
graph.fuse_ops(pattern_list)

# Initialize the graph driver and lower to top-level IR.
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()

# Set the output path prefix for MLIR files and parameters.
path_prefix = os.path.dirname(os.path.abspath(__file__))

# Write the lowered subgraph IR to 'subgraph0.mlir'.
with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)

# Write the main graph IR to 'forward.mlir'.
with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)