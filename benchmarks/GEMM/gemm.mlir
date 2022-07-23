module{
    func.func @gemm(%a : memref<2088x2048xf64>, %b : memref<2048x2048xf64>, %c : memref<2088x2048xf64>) {
      linalg.matmul 
        ins(%a, %b: memref<2088x2048xf64>, memref<2048x2048xf64>)
       outs(%c:memref<2088x2048xf64>)
      return
    }
}
