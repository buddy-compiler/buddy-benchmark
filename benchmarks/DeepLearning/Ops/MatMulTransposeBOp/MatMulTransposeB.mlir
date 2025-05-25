func.func @kernel_placeholder(%a : tensor<40x4096xf32>, %b : tensor<4096x4096xf32>) -> tensor<40x4096xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
  %ret = linalg.matmul_transpose_b 
    ins(%a, %b: tensor<40x4096xf32>, tensor<4096x4096xf32>)
    outs(%cst: tensor<40x4096xf32>) -> tensor<40x4096xf32>

  return %ret : tensor<40x4096xf32>
}
