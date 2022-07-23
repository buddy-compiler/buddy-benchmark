#map0 = affine_map<(d0) -> (d0 * 110)>
#map1 = affine_map<(d0) -> (d0 * 110 + 110, 696)>
#map2 = affine_map<(d0) -> (480, d0 * -480 + 2048)>
#map3 = affine_map<(d0, d1) -> (d0 * 3 + d1)>
#map4 = affine_map<(d0, d1) -> (d0 * 480 + d1)>
#map5 = affine_map<(d0, d1) -> (d0 * 64 + d1 * 4)>
#map6 = affine_map<(d0, d1) -> (0)>
#map7 = affine_map<(d0) -> (d0 + 1)>
#map8 = affine_map<(d0) -> (d0 + 2)>
module {
  func.func @gemm(%arg0: memref<2088x2048xf64>, %arg1: memref<2048x2048xf64>, %arg2: memref<2088x2048xf64>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg3 = 0 to 5 {
      affine.for %arg4 = 0 to 7 {
        affine.for %arg5 = 0 to 32 {
          affine.for %arg6 = #map0(%arg4) to min #map1(%arg4) {
            affine.for %arg7 = 0 to min #map2(%arg3) {
              affine.for %arg8 = 0 to 16 {
                %0 = affine.apply #map3(%arg6, %c0)
                %1 = affine.apply #map4(%arg3, %arg7)
                %2 = affine.apply #map5(%arg5, %arg8)
                %3 = vector.transfer_read %arg0[%0, %1], %cst {permutation_map = #map6} : memref<2088x2048xf64>, vector<4xf64>
                %4 = vector.transfer_read %arg1[%1, %2], %cst : memref<2048x2048xf64>, vector<4xf64>
                %5 = vector.transfer_read %arg2[%0, %2], %cst : memref<2088x2048xf64>, vector<4xf64>
                %6 = arith.mulf %3, %4 : vector<4xf64>
                %7 = arith.addf %5, %6 : vector<4xf64>
                vector.transfer_write %7, %arg2[%0, %2] : vector<4xf64>, memref<2088x2048xf64>
                %8 = affine.apply #map7(%c0)
                %9 = affine.apply #map3(%arg6, %8)
                %10 = affine.apply #map4(%arg3, %arg7)
                %11 = affine.apply #map5(%arg5, %arg8)
                %12 = vector.transfer_read %arg0[%9, %10], %cst {permutation_map = #map6} : memref<2088x2048xf64>, vector<4xf64>
                %13 = vector.transfer_read %arg1[%10, %11], %cst : memref<2048x2048xf64>, vector<4xf64>
                %14 = vector.transfer_read %arg2[%9, %11], %cst : memref<2088x2048xf64>, vector<4xf64>
                %15 = arith.mulf %12, %13 : vector<4xf64>
                %16 = arith.addf %14, %15 : vector<4xf64>
                vector.transfer_write %16, %arg2[%9, %11] : vector<4xf64>, memref<2088x2048xf64>
                %17 = affine.apply #map8(%c0)
                %18 = affine.apply #map3(%arg6, %17)
                %19 = affine.apply #map4(%arg3, %arg7)
                %20 = affine.apply #map5(%arg5, %arg8)
                %21 = vector.transfer_read %arg0[%18, %19], %cst {permutation_map = #map6} : memref<2088x2048xf64>, vector<4xf64>
                %22 = vector.transfer_read %arg1[%19, %20], %cst : memref<2048x2048xf64>, vector<4xf64>
                %23 = vector.transfer_read %arg2[%18, %20], %cst : memref<2088x2048xf64>, vector<4xf64>
                %24 = arith.mulf %21, %22 : vector<4xf64>
                %25 = arith.addf %23, %24 : vector<4xf64>
                vector.transfer_write %25, %arg2[%18, %20] : vector<4xf64>, memref<2088x2048xf64>
              }
            }
          }
        }
      }
    }
    return
  }
}

