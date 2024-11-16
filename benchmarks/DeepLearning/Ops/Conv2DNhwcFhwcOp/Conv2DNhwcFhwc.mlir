// generate from iree processed alexnet or lenet mlir file
func.func @conv_2d_nhwc_fhwc(%input: memref<1x28x28x10xf32>, %filter: memref<16x5x5x10xf32>, %output: memref<1x24x24x16xf32>) {
  linalg.conv_2d_nhwc_fhwc { dilations = dense<1> : tensor<2xi64>, 
                          strides = dense<1> : tensor<2xi64> } 
    ins(%input, %filter : memref<1x28x28x10xf32>, memref<16x5x5x10xf32>) 
    outs(%output : memref<1x24x24x16xf32>)
  return
}
