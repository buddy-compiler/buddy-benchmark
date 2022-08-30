// generate from iree processed alexnet or lenet mlir file
func.func @conv2d(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
    linalg.conv_2d_nchw_fchw 
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
    ins(%input, %filter : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) 
    outs(%output : memref<?x?x?x?xf32>)
    return
}
