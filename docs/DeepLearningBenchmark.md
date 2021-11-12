# Deep Learning Benchmark

The deep learning benchmark aims to evaluate models, layers, and operations from different frameworks. 
Currently, we support model evaluation, and the layers and operations parts are working in progress.

## Models

### MobileNet (IREE -> MLIR Core -> MLIR Toolchain)

This code generation road depends on different frameworks (i.e., IREE and TensorFlow), which are difficult to automate. 
We thus provide a model source code generated from IREE and perform the preprocessing. 
The following is the process of generating code.

**1. Build IREE and dependencies**

- Build IREE. [steps](https://google.github.io/iree/building-from-source/getting-started/)
- Build python bindings and importers. [steps](https://google.github.io/iree/building-from-source/python-bindings-and-importers/)
- Check the tools: 
    - `iree/integrations/tensorflow/bazel-bin/iree_tf_compiler/iree-import-tf`
    - `iree-build/iree/tools/iree-opt`

**2. Prepare the model**

- Download model from TensorFlow Hub. [link](https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/classification/4)
- Add signature in mode. [steps](https://google.github.io/iree/ml-frameworks/tensorflow/#missing-serving-signature-in-savedmodel)

**3. Generate mlir file and preprocess**

- Generate mlir file.

```
<iree-import-tf> -tf-import-type=savedmodel_v1 -tf-savedmodel-exported-names=predict </path/to/mobilenet/> -o <generated file>
```

- Lower to MLIR core abstraction.

```
<iree-opt> --iree-util-fold-globals <generated file> -o <output file>
```

- Preprocess

First of all, erase the `predict` function in the output file. 
And then, rename the `predict__ireesm` function and remove the attribute of the function. 
Now you can get the final model file.
