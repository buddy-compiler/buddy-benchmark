#map = affine_map<(d0) -> (256, 4096-d0)>
#map1 = affine_map<(d0) -> (64, 4096-d0)>
#map2 = affine_map<(d0) -> (8, 4096-d0)>
#map3 = affine_map<(d0) -> (4, 4096-d0)>
func.func @manul(%arg0: memref<4096x4096xf32>, %arg1: memref<4096xf32>) {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c4096 = arith.constant 4096 : index
  %c1 = arith.constant 1 : index
  affine.for %arg3 = %c0 to %c4096 step 1 {
    affine.for %arg4 = %c0 to %c4096 step 32 {
      %0 = affine.vector_load %arg0[%arg3, %arg4] : memref<4096x4096xf32>, vector<32xf32>
      %1 = affine.vector_load %arg1[%arg4] : memref<4096xf32>, vector<32xf32>
      %2 = arith.addf %0, %1 : vector<32xf32>
      affine.vector_store %2, %arg1[%arg4] : memref<4096xf32>, vector<32xf32>
    }
  }
  // affine.for %arg3 = %c0 to %c4096 step 256 {
  //   affine.for %arg4 = %c0 to %c4096 step 256 {
  //     affine.for %arg5 = %c0 to min #map(%arg3) step 1 {
  //       affine.for %arg6 = %c0 to min #map(%arg4) step 32 {
  //         // %0 = affine.load %arg0[%arg3 + %arg5, %arg4 + %arg6] : memref<4096x4096xf32>
  //         // %1 = affine.load %arg1[%arg3 + %arg5] : memref<4096xf32>
  //         // %2 = arith.addf %0, %1 : f32
  //         // affine.store %2, %arg1[%arg3 + %arg5] : memref<4096xf32>
  //         %0 = affine.vector_load %arg0[%arg3 + %arg5, %arg4 + %arg6] : memref<4096x4096xf32>, vector<32xf32>
  //         %1 = affine.vector_load %arg1[%arg4 + %arg6] : memref<4096xf32>, vector<32xf32>
  //         %2 = arith.addf %0, %1 : vector<32xf32>
  //         affine.vector_store %2, %arg1[%arg4 + %arg6] : memref<4096xf32>, vector<32xf32>
  //       }
  //     }
  //   }
  // }
  return
}

func.func @manul2(%arg0: memref<4096x4096xf32>, %arg1: memref<4096xf32>) {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c4096 = arith.constant 4096 : index
  %c1 = arith.constant 1 : index
  affine.for %arg3 = %c0 to %c4096 step 8 {
    affine.for %arg4 = %c0 to %c4096 step 1 {
      affine.for %arg5 = %c0 to min #map2(%arg3) {
        %0 = affine.load %arg0[%arg3 + %arg5, %arg4] : memref<4096x4096xf32>
        %1 = affine.load %arg1[%arg3 + %arg5] : memref<4096xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %arg1[%arg3 + %arg5] : memref<4096xf32>
      }
    } 
  }
  return
}

func.func @manul3(%arg0: memref<4096x4096xf32>, %arg1: memref<4096xf32>) {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c4096 = arith.constant 4096 : index
  %c1 = arith.constant 1 : index
  affine.for %arg3 = %c0 to %c4096 step 64 {
    affine.for %arg4 = %c0 to %c4096 step 64 {
      affine.for %arg5 = %c0 to min #map1(%arg3) step 1 {
        affine.for %arg6 = %c0 to min #map1(%arg4) step 32 {
          %0 = affine.vector_load %arg0[%arg3 + %arg5, %arg4 + %arg6] : memref<4096x4096xf32>, vector<32xf32>
          %1 = affine.vector_load %arg1[%arg4 + %arg6] : memref<4096xf32>, vector<32xf32>
          %2 = arith.addf %0, %1 : vector<32xf32>
          affine.vector_store %2, %arg1[%arg4 + %arg6] : memref<4096xf32>, vector<32xf32>
        }
      }
    }
  }
  return
}