module{
  func.func @matmul(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    linalg.matmul 
      ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
      outs(%c: memref<?x?xf32>)
    return
  }
}
