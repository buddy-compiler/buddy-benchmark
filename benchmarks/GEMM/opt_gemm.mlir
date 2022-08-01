#map0 = affine_map<()[s0] -> (s0 ceildiv 480)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 330)>
#map2 = affine_map<()[s0] -> (s0 ceildiv 16)>
#map3 = affine_map<(d0) -> (d0 * 110)>
#map4 = affine_map<(d0)[s0] -> (d0 * 110 + 110, s0 floordiv 3)>
#map5 = affine_map<(d0)[s0] -> (480, d0 * -480 + s0)>
#map6 = affine_map<(d0)[s0] -> (16, d0 * -16 + s0)>
#map7 = affine_map<(d0) -> (d0 * 3)>
#map8 = affine_map<(d0, d1) -> (d0 * 480 + d1)>
#map9 = affine_map<(d0, d1) -> (d0 * 128 + d1 * 8)>
#map10 = affine_map<(d0, d1) -> (0)>
#map11 = affine_map<(d0) -> (d0 * 3 + 1)>
#map12 = affine_map<(d0) -> (d0 * 3 + 2)>
module {
  func.func @gemm(%arg0: memref<?x?xf64>, %arg1: memref<?x?xf64>, %arg2: memref<?x?xf64>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = memref.dim %arg0, %c0 : memref<?x?xf64>
    %1 = memref.dim %arg0, %c1 : memref<?x?xf64>
    %2 = memref.dim %arg1, %c1 : memref<?x?xf64>
    affine.for %arg3 = 0 to #map0()[%1] {
      affine.for %arg4 = 0 to #map1()[%0] {
        affine.for %arg5 = 0 to #map2()[%2] {
          affine.for %arg6 = #map3(%arg4) to min #map4(%arg4)[%0] {
            affine.for %arg7 = 0 to min #map5(%arg3)[%1] {
              affine.for %arg8 = 0 to min #map6(%arg5)[%2] {
                %3 = affine.apply #map7(%arg6)
                %4 = affine.apply #map8(%arg3, %arg7)
                %5 = affine.apply #map9(%arg5, %arg8)
                %6 = vector.transfer_read %arg0[%3, %4], %cst {permutation_map = #map10} : memref<?x?xf64>, vector<8xf64>
                %7 = vector.transfer_read %arg1[%4, %5], %cst : memref<?x?xf64>, vector<8xf64>
                %8 = vector.transfer_read %arg2[%3, %5], %cst : memref<?x?xf64>, vector<8xf64>
                %9 = arith.mulf %6, %7 : vector<8xf64>
                %10 = arith.addf %8, %9 : vector<8xf64>
                vector.transfer_write %10, %arg2[%3, %5] : vector<8xf64>, memref<?x?xf64>
                %11 = affine.apply #map11(%arg6)
                %12 = affine.apply #map8(%arg3, %arg7)
                %13 = affine.apply #map9(%arg5, %arg8)
                %14 = vector.transfer_read %arg0[%11, %12], %cst {permutation_map = #map10} : memref<?x?xf64>, vector<8xf64>
                %15 = vector.transfer_read %arg1[%12, %13], %cst : memref<?x?xf64>, vector<8xf64>
                %16 = vector.transfer_read %arg2[%11, %13], %cst : memref<?x?xf64>, vector<8xf64>
                %17 = arith.mulf %14, %15 : vector<8xf64>
                %18 = arith.addf %16, %17 : vector<8xf64>
                vector.transfer_write %18, %arg2[%11, %13] : vector<8xf64>, memref<?x?xf64>
                %19 = affine.apply #map12(%arg6)
                %20 = affine.apply #map8(%arg3, %arg7)
                %21 = affine.apply #map9(%arg5, %arg8)
                %22 = vector.transfer_read %arg0[%19, %20], %cst {permutation_map = #map10} : memref<?x?xf64>, vector<8xf64>
                %23 = vector.transfer_read %arg1[%20, %21], %cst : memref<?x?xf64>, vector<8xf64>
                %24 = vector.transfer_read %arg2[%19, %21], %cst : memref<?x?xf64>, vector<8xf64>
                %25 = arith.mulf %22, %23 : vector<8xf64>
                %26 = arith.addf %24, %25 : vector<8xf64>
                vector.transfer_write %26, %arg2[%19, %21] : vector<8xf64>, memref<?x?xf64>
              }
            }
          }
        }
      }
    }
    return
  }
}

