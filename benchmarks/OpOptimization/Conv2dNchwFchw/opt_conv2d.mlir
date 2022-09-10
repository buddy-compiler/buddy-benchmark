#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 + d1)>
#map2 = affine_map<(d0, d1) -> (d0 + d1 + 16)>
#map3 = affine_map<(d0) -> (d0 + 16)>
#map4 = affine_map<(d0, d1) -> (d0 + d1 + 1)>
#map5 = affine_map<(d0) -> (d0 + 1)>
#map6 = affine_map<(d0, d1) -> (d0 + d1 + 2)>
#map7 = affine_map<(d0) -> (d0 + 2)>
#map8 = affine_map<(d0, d1) -> (d0 + d1 + 3)>
#map9 = affine_map<(d0) -> (d0 + 3)>
#map10 = affine_map<(d0) -> (d0 - d0 mod 4)>
#map11 = affine_map<(d0) -> (d0 - d0 mod 32)>
module {
  func.func @conv2d(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_0 = arith.constant 0 : index
    %0 = memref.dim %arg2, %c0_0 : memref<?x?x?x?xf32>
    %c1 = arith.constant 1 : index
    %1 = memref.dim %arg2, %c1 : memref<?x?x?x?xf32>
    %c3 = arith.constant 3 : index
    %2 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32>
    %c2 = arith.constant 2 : index
    %3 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
    %c1_1 = arith.constant 1 : index
    %4 = memref.dim %arg0, %c1_1 : memref<?x?x?x?xf32>
    %c2_2 = arith.constant 2 : index
    %5 = memref.dim %arg1, %c2_2 : memref<?x?x?x?xf32>
    %c3_3 = arith.constant 3 : index
    %6 = memref.dim %arg1, %c3_3 : memref<?x?x?x?xf32>
    %7 = memref.alloc() : memref<1xvector<16xf32>>
    affine.for %arg3 = #map0(%c0) to #map0(%0) {
      affine.for %arg4 = #map0(%c0) to #map0(%1) {
        affine.for %arg5 = #map0(%c0) to #map0(%2) {
          affine.for %arg6 = #map0(%c0) to #map0(%3) {
            %8 = vector.splat %cst : vector<16xf32>
            memref.store %8, %7[%c0] : memref<1xvector<16xf32>>
            affine.for %arg7 = #map0(%c0) to #map0(%4) {
              affine.for %arg8 = #map0(%c0) to #map0(%5) step 4 {
                affine.for %arg9 = #map0(%c0) to #map0(%6) step 32 {
                  %13 = affine.apply #map1(%arg6, %arg8)
                  %14 = affine.apply #map0(%arg8)
                  %15 = affine.apply #map1(%arg5, %arg9)
                  %16 = affine.apply #map0(%arg9)
                  %cst_4 = arith.constant 0.000000e+00 : f32
                  %17 = vector.transfer_read %arg0[%arg3, %arg7, %13, %15], %cst_4 : memref<?x?x?x?xf32>, vector<16xf32>
                  %cst_5 = arith.constant 0.000000e+00 : f32
                  %18 = vector.transfer_read %arg1[%arg4, %arg7, %14, %16], %cst_5 : memref<?x?x?x?xf32>, vector<16xf32>
                  %19 = affine.apply #map2(%arg5, %arg9)
                  %20 = affine.apply #map3(%arg9)
                  %cst_6 = arith.constant 0.000000e+00 : f32
                  %21 = vector.transfer_read %arg0[%arg3, %arg7, %13, %19], %cst_6 : memref<?x?x?x?xf32>, vector<16xf32>
                  %cst_7 = arith.constant 0.000000e+00 : f32
                  %22 = vector.transfer_read %arg1[%arg4, %arg7, %14, %20], %cst_7 : memref<?x?x?x?xf32>, vector<16xf32>
                  %23 = affine.apply #map4(%arg6, %arg8)
                  %24 = affine.apply #map5(%arg8)
                  %25 = affine.apply #map1(%arg5, %arg9)
                  %26 = affine.apply #map0(%arg9)
                  %cst_8 = arith.constant 0.000000e+00 : f32
                  %27 = vector.transfer_read %arg0[%arg3, %arg7, %23, %25], %cst_8 : memref<?x?x?x?xf32>, vector<16xf32>
                  %cst_9 = arith.constant 0.000000e+00 : f32
                  %28 = vector.transfer_read %arg1[%arg4, %arg7, %24, %26], %cst_9 : memref<?x?x?x?xf32>, vector<16xf32>
                  %29 = affine.apply #map2(%arg5, %arg9)
                  %30 = affine.apply #map3(%arg9)
                  %cst_10 = arith.constant 0.000000e+00 : f32
                  %31 = vector.transfer_read %arg0[%arg3, %arg7, %23, %29], %cst_10 : memref<?x?x?x?xf32>, vector<16xf32>
                  %cst_11 = arith.constant 0.000000e+00 : f32
                  %32 = vector.transfer_read %arg1[%arg4, %arg7, %24, %30], %cst_11 : memref<?x?x?x?xf32>, vector<16xf32>
                  %33 = affine.apply #map6(%arg6, %arg8)
                  %34 = affine.apply #map7(%arg8)
                  %35 = affine.apply #map1(%arg5, %arg9)
                  %36 = affine.apply #map0(%arg9)
                  %cst_12 = arith.constant 0.000000e+00 : f32
                  %37 = vector.transfer_read %arg0[%arg3, %arg7, %33, %35], %cst_12 : memref<?x?x?x?xf32>, vector<16xf32>
                  %cst_13 = arith.constant 0.000000e+00 : f32
                  %38 = vector.transfer_read %arg1[%arg4, %arg7, %34, %36], %cst_13 : memref<?x?x?x?xf32>, vector<16xf32>
                  %39 = affine.apply #map2(%arg5, %arg9)
                  %40 = affine.apply #map3(%arg9)
                  %cst_14 = arith.constant 0.000000e+00 : f32
                  %41 = vector.transfer_read %arg0[%arg3, %arg7, %33, %39], %cst_14 : memref<?x?x?x?xf32>, vector<16xf32>
                  %cst_15 = arith.constant 0.000000e+00 : f32
                  %42 = vector.transfer_read %arg1[%arg4, %arg7, %34, %40], %cst_15 : memref<?x?x?x?xf32>, vector<16xf32>
                  %43 = affine.apply #map8(%arg6, %arg8)
                  %44 = affine.apply #map9(%arg8)
                  %45 = affine.apply #map1(%arg5, %arg9)
                  %46 = affine.apply #map0(%arg9)
                  %cst_16 = arith.constant 0.000000e+00 : f32
                  %47 = vector.transfer_read %arg0[%arg3, %arg7, %43, %45], %cst_16 : memref<?x?x?x?xf32>, vector<16xf32>
                  %cst_17 = arith.constant 0.000000e+00 : f32
                  %48 = vector.transfer_read %arg1[%arg4, %arg7, %44, %46], %cst_17 : memref<?x?x?x?xf32>, vector<16xf32>
                  %49 = affine.apply #map2(%arg5, %arg9)
                  %50 = affine.apply #map3(%arg9)
                  %cst_18 = arith.constant 0.000000e+00 : f32
                  %51 = vector.transfer_read %arg0[%arg3, %arg7, %43, %49], %cst_18 : memref<?x?x?x?xf32>, vector<16xf32>
                  %cst_19 = arith.constant 0.000000e+00 : f32
                  %52 = vector.transfer_read %arg1[%arg4, %arg7, %44, %50], %cst_19 : memref<?x?x?x?xf32>, vector<16xf32>
                  %53 = memref.load %7[%c0] : memref<1xvector<16xf32>>
                  %54 = vector.fma %17, %18, %53 : vector<16xf32>
                  %55 = vector.fma %21, %22, %54 : vector<16xf32>
                  %56 = vector.fma %27, %28, %55 : vector<16xf32>
                  %57 = vector.fma %31, %32, %56 : vector<16xf32>
                  %58 = vector.fma %37, %38, %57 : vector<16xf32>
                  %59 = vector.fma %41, %42, %58 : vector<16xf32>
                  %60 = vector.fma %47, %48, %59 : vector<16xf32>
                  %61 = vector.fma %51, %52, %60 : vector<16xf32>
                  memref.store %61, %7[%c0] : memref<1xvector<16xf32>>
                }
              }
            }
            %9 = memref.load %7[%c0] : memref<1xvector<16xf32>>
            %10 = vector.reduction <add>, %9 : vector<16xf32> into f32
            %11 = memref.load %arg2[%arg3, %arg4, %arg6, %arg5] : memref<?x?x?x?xf32>
            %12 = arith.addf %11, %10 : f32
            memref.store %12, %arg2[%arg3, %arg4, %arg6, %arg5] : memref<?x?x?x?xf32>
            affine.for %arg7 = #map0(%c0) to #map0(%4) {
              %13 = affine.apply #map10(%5)
              affine.for %arg8 = #map0(%13) to #map0(%5) {
                %14 = affine.apply #map11(%6)
                affine.for %arg9 = #map0(%14) to #map0(%6) {
                  %15 = affine.apply #map1(%arg6, %arg8)
                  %16 = affine.apply #map1(%arg5, %arg9)
                  %17 = memref.load %arg0[%arg3, %arg7, %15, %16] : memref<?x?x?x?xf32>
                  %18 = memref.load %arg1[%arg4, %arg7, %arg8, %arg9] : memref<?x?x?x?xf32>
                  %19 = memref.load %arg2[%arg3, %arg4, %arg6, %arg5] : memref<?x?x?x?xf32>
                  %20 = arith.mulf %17, %18 : f32
                  %21 = arith.addf %20, %19 : f32
                  memref.store %21, %arg2[%arg3, %arg4, %arg6, %arg5] : memref<?x?x?x?xf32>
                }
              }
            }
          }
        }
      }
    }
    memref.dealloc %7 : memref<1xvector<16xf32>>
    return
  }
}

