#map0 = affine_map<()[s0] -> (s0 - 63)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map3 = affine_map<(d0)[s0] -> (d0 + 1, s0 - 1)>
#map4 = affine_map<(d0)[s0] -> (d0 + 2, s0 - 1)>
#map5 = affine_map<(d0)[s0] -> (d0 + 3, s0 - 1)>
#map6 = affine_map<(d0)[s0] -> (d0 + 4, s0 - 1)>
#map7 = affine_map<(d0)[s0] -> (d0 + 5, s0 - 1)>
#map8 = affine_map<(d0)[s0] -> (d0, s0 - 1)>
#map9 = affine_map<(d0, d1) -> (0)>
#map10 = affine_map<(d0) -> (d0 + 16)>
#map11 = affine_map<(d0) -> (d0 + 32)>
#map12 = affine_map<(d0) -> (d0 + 48)>
module {
  func.func @gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %0 = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %c1_1 = arith.constant 1 : index
    %1 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %c1_2 = arith.constant 1 : index
    %2 = memref.dim %arg0, %c1_2 : memref<?x?xf32>
    %3 = affine.apply #map0()[%1]
    affine.for %arg3 = #map1(%c0) to #map1(%3) step 64 {
      affine.for %arg4 = #map1(%c0) to #map1(%0) step 6 {
        %4 = memref.subview %arg0[%arg4, %c0] [%c1, %2] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        %5 = affine.min #map3(%arg4)[%0]
        %6 = memref.subview %arg0[%5, %c0] [%c1, %2] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        %7 = affine.min #map4(%arg4)[%0]
        %8 = memref.subview %arg0[%7, %c0] [%c1, %2] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        %9 = affine.min #map5(%arg4)[%0]
        %10 = memref.subview %arg0[%9, %c0] [%c1, %2] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        %11 = affine.min #map6(%arg4)[%0]
        %12 = memref.subview %arg0[%11, %c0] [%c1, %2] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        %13 = affine.min #map7(%arg4)[%0]
        %14 = memref.subview %arg0[%13, %c0] [%c1, %2] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        %15 = affine.min #map8(%arg4)[%0]
        %16 = memref.subview %arg2[%15, %c0] [%c1, %1] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        %17 = affine.min #map3(%arg4)[%0]
        %18 = memref.subview %arg2[%17, %c0] [%c1, %1] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        %19 = affine.min #map4(%arg4)[%0]
        %20 = memref.subview %arg2[%19, %c0] [%c1, %1] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        %21 = affine.min #map5(%arg4)[%0]
        %22 = memref.subview %arg2[%21, %c0] [%c1, %1] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        %23 = affine.min #map6(%arg4)[%0]
        %24 = memref.subview %arg2[%23, %c0] [%c1, %1] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        %25 = affine.min #map7(%arg4)[%0]
        %26 = memref.subview %arg2[%25, %c0] [%c1, %1] [%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, #map2>
        affine.for %arg5 = #map1(%c0) to #map1(%2) {
          %cst = arith.constant 0.000000e+00 : f32
          %27 = vector.transfer_read %4[%c0, %arg5], %cst {permutation_map = #map9} : memref<?x?xf32, #map2>, vector<16xf32>
          %cst_3 = arith.constant 0.000000e+00 : f32
          %28 = vector.transfer_read %6[%c0, %arg5], %cst_3 {permutation_map = #map9} : memref<?x?xf32, #map2>, vector<16xf32>
          %cst_4 = arith.constant 0.000000e+00 : f32
          %29 = vector.transfer_read %8[%c0, %arg5], %cst_4 {permutation_map = #map9} : memref<?x?xf32, #map2>, vector<16xf32>
          %cst_5 = arith.constant 0.000000e+00 : f32
          %30 = vector.transfer_read %10[%c0, %arg5], %cst_5 {permutation_map = #map9} : memref<?x?xf32, #map2>, vector<16xf32>
          %cst_6 = arith.constant 0.000000e+00 : f32
          %31 = vector.transfer_read %12[%c0, %arg5], %cst_6 {permutation_map = #map9} : memref<?x?xf32, #map2>, vector<16xf32>
          %cst_7 = arith.constant 0.000000e+00 : f32
          %32 = vector.transfer_read %14[%c0, %arg5], %cst_7 {permutation_map = #map9} : memref<?x?xf32, #map2>, vector<16xf32>
          %33 = affine.apply #map1(%arg3)
          %cst_8 = arith.constant 0.000000e+00 : f32
          %34 = vector.transfer_read %16[%c0, %33], %cst_8 : memref<?x?xf32, #map2>, vector<16xf32>
          %35 = affine.apply #map10(%arg3)
          %cst_9 = arith.constant 0.000000e+00 : f32
          %36 = vector.transfer_read %16[%c0, %35], %cst_9 : memref<?x?xf32, #map2>, vector<16xf32>
          %37 = affine.apply #map11(%arg3)
          %cst_10 = arith.constant 0.000000e+00 : f32
          %38 = vector.transfer_read %16[%c0, %37], %cst_10 : memref<?x?xf32, #map2>, vector<16xf32>
          %39 = affine.apply #map12(%arg3)
          %cst_11 = arith.constant 0.000000e+00 : f32
          %40 = vector.transfer_read %16[%c0, %39], %cst_11 : memref<?x?xf32, #map2>, vector<16xf32>
          %41 = affine.apply #map1(%arg3)
          %cst_12 = arith.constant 0.000000e+00 : f32
          %42 = vector.transfer_read %18[%c0, %41], %cst_12 : memref<?x?xf32, #map2>, vector<16xf32>
          %43 = affine.apply #map10(%arg3)
          %cst_13 = arith.constant 0.000000e+00 : f32
          %44 = vector.transfer_read %18[%c0, %43], %cst_13 : memref<?x?xf32, #map2>, vector<16xf32>
          %45 = affine.apply #map11(%arg3)
          %cst_14 = arith.constant 0.000000e+00 : f32
          %46 = vector.transfer_read %18[%c0, %45], %cst_14 : memref<?x?xf32, #map2>, vector<16xf32>
          %47 = affine.apply #map12(%arg3)
          %cst_15 = arith.constant 0.000000e+00 : f32
          %48 = vector.transfer_read %18[%c0, %47], %cst_15 : memref<?x?xf32, #map2>, vector<16xf32>
          %49 = affine.apply #map1(%arg3)
          %cst_16 = arith.constant 0.000000e+00 : f32
          %50 = vector.transfer_read %20[%c0, %49], %cst_16 : memref<?x?xf32, #map2>, vector<16xf32>
          %51 = affine.apply #map10(%arg3)
          %cst_17 = arith.constant 0.000000e+00 : f32
          %52 = vector.transfer_read %20[%c0, %51], %cst_17 : memref<?x?xf32, #map2>, vector<16xf32>
          %53 = affine.apply #map11(%arg3)
          %cst_18 = arith.constant 0.000000e+00 : f32
          %54 = vector.transfer_read %20[%c0, %53], %cst_18 : memref<?x?xf32, #map2>, vector<16xf32>
          %55 = affine.apply #map12(%arg3)
          %cst_19 = arith.constant 0.000000e+00 : f32
          %56 = vector.transfer_read %20[%c0, %55], %cst_19 : memref<?x?xf32, #map2>, vector<16xf32>
          %57 = affine.apply #map1(%arg3)
          %cst_20 = arith.constant 0.000000e+00 : f32
          %58 = vector.transfer_read %22[%c0, %57], %cst_20 : memref<?x?xf32, #map2>, vector<16xf32>
          %59 = affine.apply #map10(%arg3)
          %cst_21 = arith.constant 0.000000e+00 : f32
          %60 = vector.transfer_read %22[%c0, %59], %cst_21 : memref<?x?xf32, #map2>, vector<16xf32>
          %61 = affine.apply #map11(%arg3)
          %cst_22 = arith.constant 0.000000e+00 : f32
          %62 = vector.transfer_read %22[%c0, %61], %cst_22 : memref<?x?xf32, #map2>, vector<16xf32>
          %63 = affine.apply #map12(%arg3)
          %cst_23 = arith.constant 0.000000e+00 : f32
          %64 = vector.transfer_read %22[%c0, %63], %cst_23 : memref<?x?xf32, #map2>, vector<16xf32>
          %65 = affine.apply #map1(%arg3)
          %cst_24 = arith.constant 0.000000e+00 : f32
          %66 = vector.transfer_read %24[%c0, %65], %cst_24 : memref<?x?xf32, #map2>, vector<16xf32>
          %67 = affine.apply #map10(%arg3)
          %cst_25 = arith.constant 0.000000e+00 : f32
          %68 = vector.transfer_read %24[%c0, %67], %cst_25 : memref<?x?xf32, #map2>, vector<16xf32>
          %69 = affine.apply #map11(%arg3)
          %cst_26 = arith.constant 0.000000e+00 : f32
          %70 = vector.transfer_read %24[%c0, %69], %cst_26 : memref<?x?xf32, #map2>, vector<16xf32>
          %71 = affine.apply #map12(%arg3)
          %cst_27 = arith.constant 0.000000e+00 : f32
          %72 = vector.transfer_read %24[%c0, %71], %cst_27 : memref<?x?xf32, #map2>, vector<16xf32>
          %73 = affine.apply #map1(%arg3)
          %cst_28 = arith.constant 0.000000e+00 : f32
          %74 = vector.transfer_read %26[%c0, %73], %cst_28 : memref<?x?xf32, #map2>, vector<16xf32>
          %75 = affine.apply #map10(%arg3)
          %cst_29 = arith.constant 0.000000e+00 : f32
          %76 = vector.transfer_read %26[%c0, %75], %cst_29 : memref<?x?xf32, #map2>, vector<16xf32>
          %77 = affine.apply #map11(%arg3)
          %cst_30 = arith.constant 0.000000e+00 : f32
          %78 = vector.transfer_read %26[%c0, %77], %cst_30 : memref<?x?xf32, #map2>, vector<16xf32>
          %79 = affine.apply #map12(%arg3)
          %cst_31 = arith.constant 0.000000e+00 : f32
          %80 = vector.transfer_read %26[%c0, %79], %cst_31 : memref<?x?xf32, #map2>, vector<16xf32>
          %cst_32 = arith.constant 0.000000e+00 : f32
          %81 = vector.transfer_read %arg1[%arg5, %arg3], %cst_32 : memref<?x?xf32>, vector<16xf32>
          %82 = affine.apply #map10(%arg3)
          %cst_33 = arith.constant 0.000000e+00 : f32
          %83 = vector.transfer_read %arg1[%arg5, %82], %cst_33 : memref<?x?xf32>, vector<16xf32>
          %84 = affine.apply #map11(%arg3)
          %cst_34 = arith.constant 0.000000e+00 : f32
          %85 = vector.transfer_read %arg1[%arg5, %84], %cst_34 : memref<?x?xf32>, vector<16xf32>
          %86 = affine.apply #map12(%arg3)
          %cst_35 = arith.constant 0.000000e+00 : f32
          %87 = vector.transfer_read %arg1[%arg5, %86], %cst_35 : memref<?x?xf32>, vector<16xf32>
          %88 = vector.fma %27, %81, %34 : vector<16xf32>
          %89 = vector.fma %27, %83, %36 : vector<16xf32>
          %90 = vector.fma %27, %85, %38 : vector<16xf32>
          %91 = vector.fma %27, %87, %40 : vector<16xf32>
          %92 = vector.fma %28, %81, %42 : vector<16xf32>
          %93 = vector.fma %28, %83, %44 : vector<16xf32>
          %94 = vector.fma %28, %85, %46 : vector<16xf32>
          %95 = vector.fma %28, %87, %48 : vector<16xf32>
          %96 = vector.fma %29, %81, %50 : vector<16xf32>
          %97 = vector.fma %29, %83, %52 : vector<16xf32>
          %98 = vector.fma %29, %85, %54 : vector<16xf32>
          %99 = vector.fma %29, %87, %56 : vector<16xf32>
          %100 = vector.fma %30, %81, %58 : vector<16xf32>
          %101 = vector.fma %30, %83, %60 : vector<16xf32>
          %102 = vector.fma %30, %85, %62 : vector<16xf32>
          %103 = vector.fma %30, %87, %64 : vector<16xf32>
          %104 = vector.fma %31, %81, %66 : vector<16xf32>
          %105 = vector.fma %31, %83, %68 : vector<16xf32>
          %106 = vector.fma %31, %85, %70 : vector<16xf32>
          %107 = vector.fma %31, %87, %72 : vector<16xf32>
          %108 = vector.fma %32, %81, %74 : vector<16xf32>
          %109 = vector.fma %32, %83, %76 : vector<16xf32>
          %110 = vector.fma %32, %85, %78 : vector<16xf32>
          %111 = vector.fma %32, %87, %80 : vector<16xf32>
          %112 = affine.apply #map1(%arg3)
          vector.transfer_write %88, %16[%c0, %112] : vector<16xf32>, memref<?x?xf32, #map2>
          %113 = affine.apply #map10(%arg3)
          vector.transfer_write %89, %16[%c0, %113] : vector<16xf32>, memref<?x?xf32, #map2>
          %114 = affine.apply #map11(%arg3)
          vector.transfer_write %90, %16[%c0, %114] : vector<16xf32>, memref<?x?xf32, #map2>
          %115 = affine.apply #map12(%arg3)
          vector.transfer_write %91, %16[%c0, %115] : vector<16xf32>, memref<?x?xf32, #map2>
          %116 = affine.apply #map1(%arg3)
          vector.transfer_write %92, %18[%c0, %116] : vector<16xf32>, memref<?x?xf32, #map2>
          %117 = affine.apply #map10(%arg3)
          vector.transfer_write %93, %18[%c0, %117] : vector<16xf32>, memref<?x?xf32, #map2>
          %118 = affine.apply #map11(%arg3)
          vector.transfer_write %94, %18[%c0, %118] : vector<16xf32>, memref<?x?xf32, #map2>
          %119 = affine.apply #map12(%arg3)
          vector.transfer_write %95, %18[%c0, %119] : vector<16xf32>, memref<?x?xf32, #map2>
          %120 = affine.apply #map1(%arg3)
          vector.transfer_write %96, %20[%c0, %120] : vector<16xf32>, memref<?x?xf32, #map2>
          %121 = affine.apply #map10(%arg3)
          vector.transfer_write %97, %20[%c0, %121] : vector<16xf32>, memref<?x?xf32, #map2>
          %122 = affine.apply #map11(%arg3)
          vector.transfer_write %98, %20[%c0, %122] : vector<16xf32>, memref<?x?xf32, #map2>
          %123 = affine.apply #map12(%arg3)
          vector.transfer_write %99, %20[%c0, %123] : vector<16xf32>, memref<?x?xf32, #map2>
          %124 = affine.apply #map1(%arg3)
          vector.transfer_write %100, %22[%c0, %124] : vector<16xf32>, memref<?x?xf32, #map2>
          %125 = affine.apply #map10(%arg3)
          vector.transfer_write %101, %22[%c0, %125] : vector<16xf32>, memref<?x?xf32, #map2>
          %126 = affine.apply #map11(%arg3)
          vector.transfer_write %102, %22[%c0, %126] : vector<16xf32>, memref<?x?xf32, #map2>
          %127 = affine.apply #map12(%arg3)
          vector.transfer_write %103, %22[%c0, %127] : vector<16xf32>, memref<?x?xf32, #map2>
          %128 = affine.apply #map1(%arg3)
          vector.transfer_write %104, %24[%c0, %128] : vector<16xf32>, memref<?x?xf32, #map2>
          %129 = affine.apply #map10(%arg3)
          vector.transfer_write %105, %24[%c0, %129] : vector<16xf32>, memref<?x?xf32, #map2>
          %130 = affine.apply #map11(%arg3)
          vector.transfer_write %106, %24[%c0, %130] : vector<16xf32>, memref<?x?xf32, #map2>
          %131 = affine.apply #map12(%arg3)
          vector.transfer_write %107, %24[%c0, %131] : vector<16xf32>, memref<?x?xf32, #map2>
          %132 = affine.apply #map1(%arg3)
          vector.transfer_write %108, %26[%c0, %132] : vector<16xf32>, memref<?x?xf32, #map2>
          %133 = affine.apply #map10(%arg3)
          vector.transfer_write %109, %26[%c0, %133] : vector<16xf32>, memref<?x?xf32, #map2>
          %134 = affine.apply #map11(%arg3)
          vector.transfer_write %110, %26[%c0, %134] : vector<16xf32>, memref<?x?xf32, #map2>
          %135 = affine.apply #map12(%arg3)
          vector.transfer_write %111, %26[%c0, %135] : vector<16xf32>, memref<?x?xf32, #map2>
        }
      }
    }
    return
  }
}

