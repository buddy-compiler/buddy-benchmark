#map = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func @conv2d_org(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = memref.dim %arg2, %c0 : memref<?x?x?x?xf32>
    %2 = memref.dim %arg2, %c1 : memref<?x?x?x?xf32>
    %5 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
    %6 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32>

    %1 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %3 = memref.dim %arg1, %c2 : memref<?x?x?x?xf32>
    %4 = memref.dim %arg1, %c3 : memref<?x?x?x?xf32>
    affine.for %arg3 = 0 to %0 {
      affine.for %arg4 = 0 to %2 {
        affine.for %arg5 = 0 to %5 {
          affine.for %arg6 = 0 to %6 {

            affine.for %arg7 = 0 to %1 {
              affine.for %arg8 = 0 to %3 {
                affine.for %arg9 = 0 to %4 {
                  %7 = affine.apply #map(%arg5, %arg8)
                  %8 = affine.apply #map(%arg6, %arg9)
                  %9 = affine.load %arg0[%arg3, %arg7, %7, %8] : memref<?x?x?x?xf32>
                  %10 = affine.load %arg1[%arg4, %arg7, %arg8, %arg9] : memref<?x?x?x?xf32>
                  %11 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
                  %12 = arith.mulf %9, %10 : f32
                  %13 = arith.addf %11, %12 : f32
                  affine.store %13, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
                }
              }
            }
          }
        }
      }
    }
    return
  }
}

