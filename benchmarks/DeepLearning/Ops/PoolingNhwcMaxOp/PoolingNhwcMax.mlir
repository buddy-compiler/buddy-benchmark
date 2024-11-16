// generate from iree processed alexnet or lenet mlir file
func.func @pooling_nhwc_max(%a : memref<1x28x28x10xf32>, %b : memref<5x5xf32>, %c : memref<1x12x12x10xf32>) {
    linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} 
      ins(%a, %b : memref<1x28x28x10xf32>, memref<5x5xf32>) 
      outs(%c : memref<1x12x12x10xf32>)
    return
  }
