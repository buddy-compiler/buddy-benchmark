#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (d0, s0 - 1)>
#map2 = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d1 * s2 + d0 * s1 + s0 + d2)>
#map3 = affine_map<(d0)[s0] -> (d0 + 1, s0 - 1)>
#map4 = affine_map<(d0)[s0] -> (d0 + 2, s0 - 1)>
#map5 = affine_map<(d0)[s0] -> (d0 + 3, s0 - 1)>
#map6 = affine_map<(d0) -> (d0 + 16)>
#map7 = affine_map<(d0, d1) -> (-d0 + d1)>
#map8 = affine_map<(d0) -> (d0 + 1)>
#map9 = affine_map<(d0) -> (d0 + 2)>
#map10 = affine_map<(d0) -> (d0 + 3)>
#set = affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>
#set1 = affine_set<(d0) : (d0 - 16 >= 0)>
module {
  func.func @batch_matmul(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?x?xf32>
    %c1_1 = arith.constant 1 : index
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?x?xf32>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg1, %c1_3 : memref<?x?x?xf32>
    %c2 = arith.constant 2 : index
    %dim_5 = memref.dim %arg1, %c2 : memref<?x?x?xf32>
    affine.parallel (%arg3) = (0) to (%dim) {
      affine.prefetch %arg0[%arg3, %dim_2, %dim_4], read, locality<3>, data : memref<?x?x?xf32>
      affine.for %arg4 = #map(%c0) to #map(%dim_5) step 32 {
        affine.for %arg5 = #map(%c0) to #map(%dim_2) step 4 {
          %0 = affine.min #map1(%arg5)[%dim_2]
          %subview = memref.subview %arg2[%arg3, %0, %c0] [%c1, %c1, %dim_5] [%c1, %c1, %c1] : memref<?x?x?xf32> to memref<?x?x?xf32, #map2>
          %1 = affine.min #map3(%arg5)[%dim_2]
          %subview_6 = memref.subview %arg2[%arg3, %1, %c0] [%c1, %c1, %dim_5] [%c1, %c1, %c1] : memref<?x?x?xf32> to memref<?x?x?xf32, #map2>
          %2 = affine.min #map4(%arg5)[%dim_2]
          %subview_7 = memref.subview %arg2[%arg3, %2, %c0] [%c1, %c1, %dim_5] [%c1, %c1, %c1] : memref<?x?x?xf32> to memref<?x?x?xf32, #map2>
          %3 = affine.min #map5(%arg5)[%dim_2]
          %subview_8 = memref.subview %arg2[%arg3, %3, %c0] [%c1, %c1, %dim_5] [%c1, %c1, %c1] : memref<?x?x?xf32> to memref<?x?x?xf32, #map2>
          affine.for %arg6 = #map(%c0) to #map(%dim_4) {
            %4 = vector.load %arg1[%arg3, %arg6, %arg4] : memref<?x?x?xf32>, vector<16xf32>
            %5 = affine.apply #map6(%arg4)
            %6 = vector.load %arg1[%arg3, %arg6, %5] : memref<?x?x?xf32>, vector<16xf32>
            affine.if #set(%arg5)[%dim_2] {
              %10 = memref.load %arg0[%arg3, %arg5, %arg6] : memref<?x?x?xf32>
              %11 = vector.splat %10 : vector<16xf32>
              %12 = vector.load %subview[%c0, %c0, %arg4] : memref<?x?x?xf32, #map2>, vector<16xf32>
              %13 = vector.fma %11, %4, %12 : vector<16xf32>
              %14 = affine.apply #map7(%arg4, %dim_5)
              affine.if #set1(%14) {
                vector.store %13, %subview[%c0, %c0, %arg4] : memref<?x?x?xf32, #map2>, vector<16xf32>
              } else {
                %19 = vector.create_mask %14 : vector<16xi1>
                vector.maskedstore %subview[%c0, %c0, %arg4], %19, %13 : memref<?x?x?xf32, #map2>, vector<16xi1>, vector<16xf32>
              }
              %15 = affine.apply #map6(%arg4)
              %16 = vector.load %subview[%c0, %c0, %15] : memref<?x?x?xf32, #map2>, vector<16xf32>
              %17 = vector.fma %11, %6, %16 : vector<16xf32>
              %18 = affine.apply #map7(%15, %dim_5)
              affine.if #set1(%18) {
                vector.store %17, %subview[%c0, %c0, %15] : memref<?x?x?xf32, #map2>, vector<16xf32>
              } else {
                %19 = vector.create_mask %18 : vector<16xi1>
                vector.maskedstore %subview[%c0, %c0, %15], %19, %17 : memref<?x?x?xf32, #map2>, vector<16xi1>, vector<16xf32>
              }
            }
            %7 = affine.apply #map8(%arg5)
            affine.if #set(%7)[%dim_2] {
              %10 = memref.load %arg0[%arg3, %7, %arg6] : memref<?x?x?xf32>
              %11 = vector.splat %10 : vector<16xf32>
              %12 = vector.load %subview_6[%c0, %c0, %arg4] : memref<?x?x?xf32, #map2>, vector<16xf32>
              %13 = vector.fma %11, %4, %12 : vector<16xf32>
              %14 = affine.apply #map7(%arg4, %dim_5)
              affine.if #set1(%14) {
                vector.store %13, %subview_6[%c0, %c0, %arg4] : memref<?x?x?xf32, #map2>, vector<16xf32>
              } else {
                %19 = vector.create_mask %14 : vector<16xi1>
                vector.maskedstore %subview_6[%c0, %c0, %arg4], %19, %13 : memref<?x?x?xf32, #map2>, vector<16xi1>, vector<16xf32>
              }
              %15 = affine.apply #map6(%arg4)
              %16 = vector.load %subview_6[%c0, %c0, %15] : memref<?x?x?xf32, #map2>, vector<16xf32>
              %17 = vector.fma %11, %6, %16 : vector<16xf32>
              %18 = affine.apply #map7(%15, %dim_5)
              affine.if #set1(%18) {
                vector.store %17, %subview_6[%c0, %c0, %15] : memref<?x?x?xf32, #map2>, vector<16xf32>
              } else {
                %19 = vector.create_mask %18 : vector<16xi1>
                vector.maskedstore %subview_6[%c0, %c0, %15], %19, %17 : memref<?x?x?xf32, #map2>, vector<16xi1>, vector<16xf32>
              }
            }
            %8 = affine.apply #map9(%arg5)
            affine.if #set(%8)[%dim_2] {
              %10 = memref.load %arg0[%arg3, %8, %arg6] : memref<?x?x?xf32>
              %11 = vector.splat %10 : vector<16xf32>
              %12 = vector.load %subview_7[%c0, %c0, %arg4] : memref<?x?x?xf32, #map2>, vector<16xf32>
              %13 = vector.fma %11, %4, %12 : vector<16xf32>
              %14 = affine.apply #map7(%arg4, %dim_5)
              affine.if #set1(%14) {
                vector.store %13, %subview_7[%c0, %c0, %arg4] : memref<?x?x?xf32, #map2>, vector<16xf32>
              } else {
                %19 = vector.create_mask %14 : vector<16xi1>
                vector.maskedstore %subview_7[%c0, %c0, %arg4], %19, %13 : memref<?x?x?xf32, #map2>, vector<16xi1>, vector<16xf32>
              }
              %15 = affine.apply #map6(%arg4)
              %16 = vector.load %subview_7[%c0, %c0, %15] : memref<?x?x?xf32, #map2>, vector<16xf32>
              %17 = vector.fma %11, %6, %16 : vector<16xf32>
              %18 = affine.apply #map7(%15, %dim_5)
              affine.if #set1(%18) {
                vector.store %17, %subview_7[%c0, %c0, %15] : memref<?x?x?xf32, #map2>, vector<16xf32>
              } else {
                %19 = vector.create_mask %18 : vector<16xi1>
                vector.maskedstore %subview_7[%c0, %c0, %15], %19, %17 : memref<?x?x?xf32, #map2>, vector<16xi1>, vector<16xf32>
              }
            }
            %9 = affine.apply #map10(%arg5)
            affine.if #set(%9)[%dim_2] {
              %10 = memref.load %arg0[%arg3, %9, %arg6] : memref<?x?x?xf32>
              %11 = vector.splat %10 : vector<16xf32>
              %12 = vector.load %subview_8[%c0, %c0, %arg4] : memref<?x?x?xf32, #map2>, vector<16xf32>
              %13 = vector.fma %11, %4, %12 : vector<16xf32>
              %14 = affine.apply #map7(%arg4, %dim_5)
              affine.if #set1(%14) {
                vector.store %13, %subview_8[%c0, %c0, %arg4] : memref<?x?x?xf32, #map2>, vector<16xf32>
              } else {
                %19 = vector.create_mask %14 : vector<16xi1>
                vector.maskedstore %subview_8[%c0, %c0, %arg4], %19, %13 : memref<?x?x?xf32, #map2>, vector<16xi1>, vector<16xf32>
              }
              %15 = affine.apply #map6(%arg4)
              %16 = vector.load %subview_8[%c0, %c0, %15] : memref<?x?x?xf32, #map2>, vector<16xf32>
              %17 = vector.fma %11, %6, %16 : vector<16xf32>
              %18 = affine.apply #map7(%15, %dim_5)
              affine.if #set1(%18) {
                vector.store %17, %subview_8[%c0, %c0, %15] : memref<?x?x?xf32, #map2>, vector<16xf32>
              } else {
                %19 = vector.create_mask %18 : vector<16xi1>
                vector.maskedstore %subview_8[%c0, %c0, %15], %19, %17 : memref<?x?x?xf32, #map2>, vector<16xi1>, vector<16xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}

