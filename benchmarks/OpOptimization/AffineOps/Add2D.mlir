func.func @add2d(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %N = memref.dim %A, %c1 : memref<?x?xf32>
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  affine.for %i4 = 0 to %M {
    affine.for %i5 = 0 to %N {
      %a5 = affine.load %A[%i4, %i5] : memref<?x?xf32>
      %b5 = affine.load %B[%i4, %i5] : memref<?x?xf32>
      %s5 = arith.addf %a5, %b5 : f32
      affine.store %s5, %C[%i4, %i5] : memref<?x?xf32>
    }
  }
  return
}
