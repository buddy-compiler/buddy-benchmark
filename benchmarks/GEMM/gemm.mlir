module{
    func.func @gemm(%a : memref<?x?xf64>, %b : memref<?x?xf64>, %c : memref<?x?xf64>) {
      linalg.matmul 
        ins(%a, %b: memref<?x?xf64>, memref<?x?xf64>)
       outs(%c:memref<?x?xf64>)
      return
    }
}
