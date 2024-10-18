module{
  func.func @matmul(%a : memref<?x?xi32>, %b : memref<?x?xi32>, %c : memref<?x?xi32>) {
    linalg.matmul 
      ins(%a, %b: memref<?x?xi32>, memref<?x?xi32>)
      outs(%c: memref<?x?xi32>)
    return
  }
}
