func.func @reduction(%in: memref<?x?xf32>, %out: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %in, %c0 : memref<?x?xf32>
  %N = memref.dim %in, %c1 : memref<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i = 0 to %M {
    %final_red = affine.for %j = 0 to %N iter_args(%red_iter = %cst) -> (f32) {
      %ld = affine.load %in[%i, %j] : memref<?x?xf32>
      %add = arith.addf %red_iter, %ld : f32
      affine.yield %add : f32
    }
    affine.store %final_red, %out[%i] : memref<?xf32>
  }
  return
}
