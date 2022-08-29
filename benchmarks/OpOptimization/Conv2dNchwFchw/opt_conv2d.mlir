#map0 = affine_map<(d0, d1, d2, d3) -> (0)>
#map1 = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func @conv2d(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
    %1 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %2 = memref.dim %arg1, %c0 : memref<?x?x?x?xf32>
    %3 = memref.dim %arg1, %c2 : memref<?x?x?x?xf32>
    %4 = memref.dim %arg1, %c3 : memref<?x?x?x?xf32>
    %5 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
    %6 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32>
    %7 = memref.alloc() : memref<1xvector<16xf32>>
    affine.for %arg3 = 0 to %0 {
      affine.for %arg4 = 0 to %2 {
        affine.for %arg5 = 0 to %6 {
          affine.for %arg6 = 0 to %5 {
            %8 = vector.transfer_read %arg2[%arg3, %arg4, %arg6, %arg5], %cst {permutation_map = #map0} : memref<?x?x?x?xf32>, vector<16xf32>
            memref.store %8, %7[%c0] : memref<1xvector<16xf32>>
            affine.for %arg7 = 0 to %1 {
              affine.for %arg8 = 0 to %3 {
                %11 = affine.apply #map1(%arg6, %arg8)
                %12 = vector.transfer_read %arg0[%arg3, %arg7, %11, %c0], %cst : memref<?x?x?x?xf32>, vector<16xf32>
                %13 = vector.transfer_read %arg1[%arg4, %arg7, %arg8, %c0], %cst : memref<?x?x?x?xf32>, vector<16xf32>
                %14 = memref.load %7[%c0] : memref<1xvector<16xf32>>
                %15 = vector.fma %12, %13, %14 : vector<16xf32>
                memref.store %15, %7[%c0] : memref<1xvector<16xf32>>
              }
            }
            %9 = memref.load %7[%c0] : memref<1xvector<16xf32>>
            %10 = vector.reduction <add>, %9 : vector<16xf32> into f32
            affine.store %10, %arg2[%arg3, %arg4, %arg6, %arg5] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return
  }
}

