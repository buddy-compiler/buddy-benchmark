func @depthwise_conv2d_nhw(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
  linalg.depthwise_conv2D_nhw 
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
    ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?xf32>) 
    outs(%arg2 : memref<?x?x?x?xf32>) 
  return
}
