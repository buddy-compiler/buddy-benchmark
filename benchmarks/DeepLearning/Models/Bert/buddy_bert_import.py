# ===- import-bert.py ----------------------------------------------------------
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
# This is the BERT model importer.
#
# ===---------------------------------------------------------------------------

import os
from pathlib import Path
import numpy as np
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import BertForSequenceClassification, BertTokenizer

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    "bhadresh-savani/bert-base-uncased-emotion"
)
model = model.eval()

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Tokenize input text for the BERT model
tokenizer = BertTokenizer.from_pretrained(
    "bhadresh-savani/bert-base-uncased-emotion"
)
inputs = tokenizer("This is a test sentence.", return_tensors="pt")
inputs = {k: v.to(torch.int64) for k, v in inputs.items()}

# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, **inputs)

# Ensure the graph has been properly imported
assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]

# Apply simple fusion transformation
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)

# Create a GraphDriver to handle the conversion to MLIR
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()

# Define the output path prefix
path_prefix = os.path.dirname(os.path.abspath(__file__))

# Save the subgraph and forward graph as MLIR files
with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)
