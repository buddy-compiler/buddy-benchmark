#map0 = affine_map<(d0) -> (d0 * 480)>
#map1 = affine_map<(d0) -> (d0 * 480 + 480, 2048)>
#map2 = affine_map<(d0, d1) -> (d1 * 330)>
#map3 = affine_map<(d0, d1) -> (d0 * 480)>
#map4 = affine_map<(d0) -> (d0 * 330)>
#map5 = affine_map<(d0) -> (d0 * 330 + 330, 2088)>
#map6 = affine_map<(d0, d1) -> (d0 * 16)>
#map7 = affine_map<(d0) -> (480, d0 * -480 + 2048)>
#map8 = affine_map<(d0) -> (d0 * 16)>
#map9 = affine_map<(d0) -> (d0 * 16 + 16)>
#map10 = affine_map<(d0) -> (d0 * 110)>
#map11 = affine_map<(d0) -> (d0 * 110 + 110, 696)>
#map12 = affine_map<(d0, d1) -> (d0 * 3 + d1)>
#map13 = affine_map<(d0, d1) -> (d0 * 16 + d1)>
#map14 = affine_map<(d0, d1) -> (d0 * 480 + d1)>
#map15 = affine_map<(d0, d1, d2) -> (d0)>
#map16 = affine_map<(d0, d1, d2) -> (d1 * 16 + d2)>
module {
  func.func @gemm(%arg0: memref<2088x2048xf64>, %arg1: memref<2048x2048xf64>, %arg2: memref<2088x2048xf64>) {
    %c7680 = arith.constant 7680 : index
    %c0 = arith.constant 0 : index
    %c158400 = arith.constant 158400 : index
    %c0_0 = arith.constant 0 : index
    %c983040 = arith.constant 983040 : index
    %c0_1 = arith.constant 0 : index
    affine.for %arg3 = 0 to 5 {
      %0 = affine.apply #map0(%arg3)
      %1 = memref.alloc() : memref<480x2048xf64>
      affine.for %arg4 = #map0(%arg3) to min #map1(%arg3) {
        affine.for %arg5 = 0 to 2048 {
          %2 = affine.load %arg1[%arg4, %arg5] : memref<2048x2048xf64>
          affine.store %2, %1[%arg4 - %arg3 * 480, %arg5] : memref<480x2048xf64>
        }
      }
      affine.for %arg4 = 0 to 7 {
        %2 = affine.apply #map2(%arg3, %arg4)
        %3 = affine.apply #map3(%arg3, %arg4)
        %4 = memref.alloc() : memref<330x480xf64>
        affine.for %arg5 = #map4(%arg4) to min #map5(%arg4) {
          affine.for %arg6 = #map0(%arg3) to min #map1(%arg3) {
            %5 = affine.load %arg0[%arg5, %arg6] : memref<2088x2048xf64>
            affine.store %5, %4[%arg5 - %arg4 * 330, %arg6 - %arg3 * 480] : memref<330x480xf64>
          }
        }
        affine.for %arg5 = 0 to 128 {
          %5 = affine.apply #map6(%arg5, %arg3)
          %6 = memref.alloc() : memref<480x16xf64>
          affine.for %arg6 = 0 to min #map7(%arg3) {
            affine.for %arg7 = #map8(%arg5) to #map9(%arg5) {
              %7 = affine.load %1[%arg6, %arg7] : memref<480x2048xf64>
              affine.store %7, %6[%arg6, %arg7 - %arg5 * 16] : memref<480x16xf64>
            }
          }
          affine.for %arg6 = #map10(%arg4) to min #map11(%arg4) {
            affine.for %arg7 = 0 to min #map7(%arg3) {
              affine.for %arg8 = 0 to 16 {
                affine.for %arg9 = 0 to 3 {
                  %7 = affine.apply #map12(%arg6, %arg9)
                  %8 = affine.apply #map13(%arg5, %arg8)
                  %9 = affine.apply #map14(%arg3, %arg7)
                  %10 = affine.load %4[%arg4 * -330 + %arg6 * 3 + %arg9, %arg7] : memref<330x480xf64>
                  %11 = affine.apply #map15(%arg7, %arg5, %arg8)
                  %12 = affine.apply #map16(%arg7, %arg5, %arg8)
                  %13 = affine.load %6[%arg7, %arg8] : memref<480x16xf64>
                  %14 = affine.load %arg2[%7, %8] : memref<2088x2048xf64>
                  %15 = arith.mulf %10, %13 : f64
                  %16 = arith.addf %14, %15 : f64
                  affine.store %16, %arg2[%7, %8] : memref<2088x2048xf64>
                }
              }
            }
          }
          memref.dealloc %6 : memref<480x16xf64>
        }
        memref.dealloc %4 : memref<330x480xf64>
      }
      memref.dealloc %1 : memref<480x2048xf64>
    }
    return
  }
}

