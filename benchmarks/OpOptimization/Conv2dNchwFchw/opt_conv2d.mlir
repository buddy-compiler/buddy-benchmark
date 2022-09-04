#map0 = affine_map<(d0, d1, d2, d3) -> (0)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
#map2 = affine_map<(d0, d1) -> (d0 + d1)>
#map3 = affine_map<(d0) -> (d0 mod 16)>
#map4 = affine_map<(d0, d1) -> (d0 - d1)>
#set = affine_set<(d0) : (d0 mod 16 - 1 >= 0)>
module {
  func.func @conv2d(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = vector.splat %cst : vector<16xf32>
    %1 = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
    %2 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %3 = memref.dim %arg1, %c0 : memref<?x?x?x?xf32>
    %4 = memref.dim %arg1, %c2 : memref<?x?x?x?xf32>
    %5 = memref.dim %arg1, %c3 : memref<?x?x?x?xf32>
    %6 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
    %7 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32>
    %8 = memref.alloc() : memref<1xvector<16xf32>>
    affine.for %arg3 = 0 to %1 {
      affine.for %arg4 = 0 to %3 {
        affine.for %arg5 = 0 to %7 {
          affine.for %arg6 = 0 to %6 {
            %9 = vector.transfer_read %arg2[%arg3, %arg4, %arg6, %arg5], %cst {permutation_map = #map0} : memref<?x?x?x?xf32>, vector<16xf32>
            memref.store %9, %8[%c0] : memref<1xvector<16xf32>>
            affine.for %arg7 = 0 to %2 {
              affine.for %arg8 = 0 to %4 step 2 {
                affine.for %arg9 = 0 to %5 step 16 {
                  %12 = affine.apply #map1(%arg8)[%c1]
                  %13 = affine.apply #map2(%arg6, %arg8)
                  %14 = affine.apply #map2(%arg6, %12)
                  %15 = affine.apply #map2(%arg5, %arg9)
                  %16 = vector.transfer_read %arg0[%arg3, %arg7, %13, %15], %cst : memref<?x?x?x?xf32>, vector<16xf32>
                  %17 = vector.transfer_read %arg1[%arg4, %arg7, %arg8, %arg9], %cst : memref<?x?x?x?xf32>, vector<16xf32>
                  %18 = vector.transfer_read %arg0[%arg3, %arg7, %14, %15], %cst : memref<?x?x?x?xf32>, vector<16xf32>
                  %19 = vector.transfer_read %arg1[%arg4, %arg7, %12, %arg9], %cst : memref<?x?x?x?xf32>, vector<16xf32>
                  %20 = memref.load %8[%c0] : memref<1xvector<16xf32>>
                  %21 = vector.fma %16, %17, %20 : vector<16xf32>
                  %22 = vector.fma %18, %19, %21 : vector<16xf32>
                  memref.store %22, %8[%c0] : memref<1xvector<16xf32>>
                }
                affine.if #set(%5) {
                  %12 = affine.apply #map3(%5)
                  %13 = vector.create_mask %12 : vector<16xi1>
                  %14 = affine.apply #map2(%arg6, %arg8)
                  %15 = affine.apply #map4(%5, %12)
                  %16 = affine.apply #map2(%arg5, %15)
                  %17 = vector.maskedload %arg0[%arg3, %arg7, %14, %16], %13, %0 : memref<?x?x?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
                  %18 = vector.maskedload %arg1[%arg4, %arg7, %arg8, %15], %13, %0 : memref<?x?x?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
                  %19 = memref.load %8[%c0] : memref<1xvector<16xf32>>
                  %20 = vector.fma %17, %18, %19 : vector<16xf32>
                  memref.store %20, %8[%c0] : memref<1xvector<16xf32>>
                }
              }
            }
            %10 = memref.load %8[%c0] : memref<1xvector<16xf32>>
            %11 = vector.reduction <add>, %10 : vector<16xf32> into f32
            affine.store %11, %arg2[%arg3, %arg4, %arg6, %arg5] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    memref.dealloc %8 : memref<1xvector<16xf32>>
    return
  }
}

