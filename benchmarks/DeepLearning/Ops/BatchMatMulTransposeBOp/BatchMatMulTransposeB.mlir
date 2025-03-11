func.func @kernel_placeholder(%a : memref<?x?x?xf32>, %b : memref<?x?x?xf32>, %c : memref<?x?x?xf32>) {
  linalg.batch_matmul_transpose_b 
    ins(%a, %b: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs(%c: memref<?x?x?xf32>)
    return
}
