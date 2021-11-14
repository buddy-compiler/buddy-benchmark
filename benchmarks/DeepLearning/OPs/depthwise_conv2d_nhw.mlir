func @depthwise_conv2D_nhw(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
  linalg.depthwise_conv2D_nhw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%input, %filter: memref<?x?x?x?xf32>, memref<?x?x?xf32>)
    outs(%output: memref<?x?x?x?xf32>)
  return
}