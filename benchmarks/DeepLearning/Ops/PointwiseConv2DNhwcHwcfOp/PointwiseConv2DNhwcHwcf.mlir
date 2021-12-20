// Generated from Mobilenet.mlir file
func @pointwise_conv_2d_nhwc_hwcf_with_return_origin(%input: tensor<1x4x5x2xf32>, %filter: tensor<1x1x2x7xf32>) -> tensor<1x4x5x7xf32> {
    %0 = linalg.init_tensor [1, 4, 5, 7] : tensor<1x4x5x7xf32>
    %1 = linalg.conv_2d_nhwc_hwcf {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x4x5x2xf32>, tensor<1x1x2x7xf32>) outs(%0 : tensor<1x4x5x7xf32>) -> tensor<1x4x5x7xf32>
    return %1 : tensor<1x4x5x7xf32>
}

func @pointwise_conv_2d_nhwc_hwcf_with_return(%arg0: tensor<1x4x5x2xf32>, %arg1: tensor<1x1x2x7xf32>) -> tensor<1x4x5x7xf32> {
    %0 = linalg.init_tensor [1, 4, 5, 7] : tensor<1x4x5x7xf32>
    %1 = tensor.collapse_shape %arg0 [[0, 1, 2], [3]] : tensor<1x4x5x2xf32> into tensor<20x2xf32>
    %2 = tensor.collapse_shape %arg1 [[0, 1, 2], [3]] : tensor<1x1x2x7xf32> into tensor<2x7xf32>
    %3 = tensor.collapse_shape %0 [[0, 1, 2], [3]] : tensor<1x4x5x7xf32> into tensor<20x7xf32>
    %4 = linalg.matmul ins(%1, %2 : tensor<20x2xf32>, tensor<2x7xf32>) outs(%3 : tensor<20x7xf32>) -> tensor<20x7xf32>
    %5 = tensor.expand_shape %4 [[0, 1, 2], [3]] : tensor<20x7xf32> into tensor<1x4x5x7xf32>
    return %5 : tensor<1x4x5x7xf32>
}

// generate from iree processed mobilenet mlir file
func @pointwise_conv_2d_nhwc_hwcf(%input: memref<?x?x?x?xf32>, %filter: memref<1x1x?x?xf32>, %output: memref<?x?x?x?xf32>) {
    linalg.conv_2d_nhwc_hwcf 
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
    ins(%input, %filter : memref<?x?x?x?xf32>, memref<1x1x?x?xf32>) 
    outs(%output : memref<?x?x?x?xf32>) 
    return
}

// test for specific shape
// func @pointwise_conv_2d_nhwc_hwcf(%input: memref<1x4x5x2xf32>, %filter: memref<1x1x2x7xf32>, %output: memref<1x4x5x7xf32>) {
//     linalg.conv_2d_nhwc_hwcf 
//     {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
//     ins(%input, %filter : memref<1x4x5x2xf32>, memref<1x1x2x7xf32>) 
//     outs(%output : memref<1x4x5x7xf32>) 
//     return
// }

// func @pointwise_conv_2d_nhwc_hwcf(%output: tensor<1x4x5x7xf32>, %input: tensor<1x4x5x2xf32>, %filter: tensor<1x1x2x7xf32>) {
//     linalg.conv_2d_nhwc_hwcf 
//     {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
//     ins(%input, %filter : tensor<1x4x5x2xf32>, tensor<1x1x2x7xf32>) 
//     outs(%output : tensor<1x4x5x7xf32>) 
//     return
// }
