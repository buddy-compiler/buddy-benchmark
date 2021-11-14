// generate from iree mhlo.convolution when c&f dim equals to 1
func @conv_2d_input_nchw_filter_fchw(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>)
  -> memref<?x?x?x?x?xf32> {
    linalg.depthwise_conv2D_nhwc 
    { 
        dilations = dense<1> : tensor<2xi64>, 
        strides = dense<1> : tensor<2xi64>
    } 
        ins(%0, %1 : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) 
        outs(%3 : memref<?x?x?x?x?xf32>) -> memref<?x?x?x?x?xf32>
    return
}

