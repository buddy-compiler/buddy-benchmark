func @pooling_nhwc_sum(%input: memref<?x?x?x?xf32>, %filter: memref<?x?xf32>,
                       %output: memref<?x?x?x?xf32>) {

  linalg.pooling_nhwc_sum
    {strides = dense<1>: tensor<2xi64>, dilations = dense<1>: tensor<2xi64>}
    ins(%input, %filter: memref<?x?x?x?xf32>, memref<?x?xf32>)
    outs(%output: memref<?x?x?x?xf32>)

  return
}
