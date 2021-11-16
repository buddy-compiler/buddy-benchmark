// generate from iree processed mobilenet mlir file
func @conv_2d_input_nhwc_filter_hwcf(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) 
{
    linalg.conv_2d_nhwc_hwcf 
    {
        dilations = dense<1> : tensor<2xi64>, 
        strides = dense<1> : tensor<2xi64>
    } 
    ins(%input, %filter : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) 
    outs(%output : memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
    return
}