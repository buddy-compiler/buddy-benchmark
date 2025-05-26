func.func @transpose_2d_placeholder(%t0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %idx = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %t1 = tosa.transpose %t0, %idx : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>

  return %t1 : tensor<?x?xf32>
}
