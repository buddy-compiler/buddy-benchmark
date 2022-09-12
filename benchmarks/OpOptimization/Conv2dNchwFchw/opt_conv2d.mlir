#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ((d0 ceildiv 2) * 2)>
#map2 = affine_map<(d0, d1) -> (d0 + d1)>
#map3 = affine_map<(d0, d1) -> (d0 + d1 + 16)>
#map4 = affine_map<(d0) -> (d0 + 16)>
#map5 = affine_map<(d0, d1) -> (d0 + d1 + 32)>
#map6 = affine_map<(d0) -> (d0 + 32)>
#map7 = affine_map<(d0, d1) -> (d0 + d1 + 1)>
#map8 = affine_map<(d0) -> (d0 + 1)>
#set = affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>
module {
  func.func @conv2d(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_0 = arith.constant 0 : index
    %0 = memref.dim %arg2, %c0_0 : memref<?x?x?x?xf32>
    %c1 = arith.constant 1 : index
    %1 = memref.dim %arg2, %c1 : memref<?x?x?x?xf32>
    %c3_1 = arith.constant 3 : index
    %2 = memref.dim %arg2, %c3_1 : memref<?x?x?x?xf32>
    %c2_2 = arith.constant 2 : index
    %3 = memref.dim %arg2, %c2_2 : memref<?x?x?x?xf32>
    %c1_3 = arith.constant 1 : index
    %4 = memref.dim %arg0, %c1_3 : memref<?x?x?x?xf32>
    %c2_4 = arith.constant 2 : index
    %5 = memref.dim %arg1, %c2_4 : memref<?x?x?x?xf32>
    %c3_5 = arith.constant 3 : index
    %6 = memref.dim %arg1, %c3_5 : memref<?x?x?x?xf32>
    %7 = memref.alloc() : memref<1xvector<16xf32>>
    affine.for %arg3 = #map0(%c0) to #map0(%0) {
      affine.for %arg4 = #map0(%c0) to #map0(%1) {
        affine.for %arg5 = #map0(%c0) to #map0(%2) {
          affine.for %arg6 = #map0(%c0) to #map0(%3) {
            %8 = vector.splat %cst : vector<16xf32>
            memref.store %8, %7[%c0] : memref<1xvector<16xf32>>
            affine.for %arg7 = #map0(%c0) to #map0(%4) {
              %13 = affine.apply #map1(%5)
              affine.for %arg8 = #map0(%c0) to #map0(%13) step 2 {
                affine.for %arg9 = #map0(%c0) to #map0(%6) step 48 {
                  %14 = affine.apply #map2(%arg6, %arg8)
                  %15 = affine.apply #map0(%arg8)
                  %16 = affine.apply #map2(%arg5, %arg9)
                  %17 = affine.apply #map0(%arg9)
                  %cst_6 = arith.constant 0.000000e+00 : f32
                  %18 = vector.transfer_read %arg0[%arg3, %arg7, %14, %16], %cst_6 : memref<?x?x?x?xf32>, vector<16xf32>
                  %19 = affine.if #set(%15)[%5] -> vector<16xf32> {
                    %cst_12 = arith.constant 0.000000e+00 : f32
                    %49 = vector.transfer_read %arg1[%arg4, %arg7, %15, %17], %cst_12 : memref<?x?x?x?xf32>, vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  } else {
                    %49 = vector.splat %cst : vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  }
                  %20 = affine.apply #map3(%arg5, %arg9)
                  %21 = affine.apply #map4(%arg9)
                  %cst_7 = arith.constant 0.000000e+00 : f32
                  %22 = vector.transfer_read %arg0[%arg3, %arg7, %14, %20], %cst_7 : memref<?x?x?x?xf32>, vector<16xf32>
                  %23 = affine.if #set(%15)[%5] -> vector<16xf32> {
                    %cst_12 = arith.constant 0.000000e+00 : f32
                    %49 = vector.transfer_read %arg1[%arg4, %arg7, %15, %21], %cst_12 : memref<?x?x?x?xf32>, vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  } else {
                    %49 = vector.splat %cst : vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  }
                  %24 = affine.apply #map5(%arg5, %arg9)
                  %25 = affine.apply #map6(%arg9)
                  %cst_8 = arith.constant 0.000000e+00 : f32
                  %26 = vector.transfer_read %arg0[%arg3, %arg7, %14, %24], %cst_8 : memref<?x?x?x?xf32>, vector<16xf32>
                  %27 = affine.if #set(%15)[%5] -> vector<16xf32> {
                    %cst_12 = arith.constant 0.000000e+00 : f32
                    %49 = vector.transfer_read %arg1[%arg4, %arg7, %15, %25], %cst_12 : memref<?x?x?x?xf32>, vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  } else {
                    %49 = vector.splat %cst : vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  }
                  %28 = affine.apply #map7(%arg6, %arg8)
                  %29 = affine.apply #map8(%arg8)
                  %30 = affine.apply #map2(%arg5, %arg9)
                  %31 = affine.apply #map0(%arg9)
                  %cst_9 = arith.constant 0.000000e+00 : f32
                  %32 = vector.transfer_read %arg0[%arg3, %arg7, %28, %30], %cst_9 : memref<?x?x?x?xf32>, vector<16xf32>
                  %33 = affine.if #set(%29)[%5] -> vector<16xf32> {
                    %cst_12 = arith.constant 0.000000e+00 : f32
                    %49 = vector.transfer_read %arg1[%arg4, %arg7, %29, %31], %cst_12 : memref<?x?x?x?xf32>, vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  } else {
                    %49 = vector.splat %cst : vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  }
                  %34 = affine.apply #map3(%arg5, %arg9)
                  %35 = affine.apply #map4(%arg9)
                  %cst_10 = arith.constant 0.000000e+00 : f32
                  %36 = vector.transfer_read %arg0[%arg3, %arg7, %28, %34], %cst_10 : memref<?x?x?x?xf32>, vector<16xf32>
                  %37 = affine.if #set(%29)[%5] -> vector<16xf32> {
                    %cst_12 = arith.constant 0.000000e+00 : f32
                    %49 = vector.transfer_read %arg1[%arg4, %arg7, %29, %35], %cst_12 : memref<?x?x?x?xf32>, vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  } else {
                    %49 = vector.splat %cst : vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  }
                  %38 = affine.apply #map5(%arg5, %arg9)
                  %39 = affine.apply #map6(%arg9)
                  %cst_11 = arith.constant 0.000000e+00 : f32
                  %40 = vector.transfer_read %arg0[%arg3, %arg7, %28, %38], %cst_11 : memref<?x?x?x?xf32>, vector<16xf32>
                  %41 = affine.if #set(%29)[%5] -> vector<16xf32> {
                    %cst_12 = arith.constant 0.000000e+00 : f32
                    %49 = vector.transfer_read %arg1[%arg4, %arg7, %29, %39], %cst_12 : memref<?x?x?x?xf32>, vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  } else {
                    %49 = vector.splat %cst : vector<16xf32>
                    affine.yield %49 : vector<16xf32>
                  }
                  %42 = memref.load %7[%c0] : memref<1xvector<16xf32>>
                  %43 = vector.fma %18, %19, %42 : vector<16xf32>
                  %44 = vector.fma %22, %23, %43 : vector<16xf32>
                  %45 = vector.fma %26, %27, %44 : vector<16xf32>
                  %46 = vector.fma %32, %33, %45 : vector<16xf32>
                  %47 = vector.fma %36, %37, %46 : vector<16xf32>
                  %48 = vector.fma %40, %41, %47 : vector<16xf32>
                  memref.store %48, %7[%c0] : memref<1xvector<16xf32>>
                }
              }
            }
            %9 = memref.load %7[%c0] : memref<1xvector<16xf32>>
            %10 = vector.reduction <add>, %9 : vector<16xf32> into f32
            %11 = memref.load %arg2[%arg3, %arg4, %arg6, %arg5] : memref<?x?x?x?xf32>
            %12 = arith.addf %11, %10 : f32
            memref.store %12, %arg2[%arg3, %arg4, %arg6, %arg5] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    memref.dealloc %7 : memref<1xvector<16xf32>>
    return
  }
}

