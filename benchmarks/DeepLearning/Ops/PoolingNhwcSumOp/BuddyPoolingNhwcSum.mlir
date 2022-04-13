func @buddy_pooling_nhwc_sum(%input: memref<1x1024x1024x1xf32>, %filter: memref<3x3xf32>,
                       %output: memref<1x1022x1022x1xf32>) {

  linalg.pooling_nhwc_sum
    {strides = dense<1>: tensor<2xi64>, dilations = dense<1>: tensor<2xi64>}
    ins(%input, %filter: memref<1x1024x1024x1xf32>, memref<3x3xf32>)
    outs(%output: memref<1x1022x1022x1xf32>)

  return
}
