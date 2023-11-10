module {
  func.func @base(%arg0: memref<1x512x4096xf32>, %arg1: memref<1x512x1024xf32>) {
    %cst = arith.constant 4.096000e+03 : f32
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4096 {
          affine.for %arg5 = 0 to 1024 {
            %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<1x512x4096xf32>
            %1 = affine.load %arg1[%arg2, %arg3, %arg5] : memref<1x512x1024xf32>
            %2 = arith.divf %0, %cst : f32
            %3 = arith.addf %2, %1 : f32
            affine.store %3, %arg1[%arg2, %arg3, %arg5] : memref<1x512x1024xf32>
          }
        }
      }
    }
    return
  }
}



#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0) -> (512-d0, 32)>
#map3 = affine_map<(d0) -> (4096-d0, 32)>
// #map3 = affine_map<(d0) -> (1024-d0, 32)>
module {
  func.func @base(%arg0: memref<1x512x4096xf32>, %arg1: memref<1x512x1024xf32>) {
    %cst = arith.constant 4.096000e+03 : f32
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    affine.for %arg2 = %c0 to %c1 step 1 {
      affine.for %arg3 = %c0 to %c512 step 32 {
        affine.for %arg4 = %c0 to %c4096 step 32 {
          affine.for %arg5 = %c0 to %c1024 step 32 {
            // %subview = memref.subview %arg0[0, %arg2, %arg3] [1, 32, 32] [1, 1, 1] : memref<1x512x4096xf32> to memref<1x32x32xf32, strided<[2097152, 4096, 1], offset: ?>>
            // %subview_0 = memref.subview %arg1[0, %arg2, %arg4] [1, 32, 32] [1, 1, 1] : memref<1x512x1024xf32> to memref<1x32x32xf32, strided<[524288, 1024, 1], offset: ?>>
            affine.for %arg6 = %c0 to min #map2(%arg3) step 1 {
              affine.for %arg7 = %c0 to min #map3(%arg4) step 
            }
            // linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%subview : memref<1x32x32xf32, strided<[2097152, 4096, 1], offset: ?>>) outs(%subview_0 : memref<1x32x32xf32, strided<[524288, 1024, 1], offset: ?>>) {
            // ^bb0(%in: f32, %out: f32):
            //   %0 = arith.divf %in, %cst : f32
            //   %1 = arith.addf %0, %out : f32
            //   linalg.yield %1 : f32
            // }
          }
        }
      }
    }
    return
  }
}