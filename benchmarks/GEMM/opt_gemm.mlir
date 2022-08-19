#map0 = affine_map<()[s0] -> (s0 ceildiv 3200)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 400)>
#map2 = affine_map<()[s0] -> (s0 ceildiv 16)>
#map3 = affine_map<(d0) -> (d0 * 400)>
#map4 = affine_map<(d0)[s0] -> (d0 * 400 + 400, s0 floordiv 8)>
#map5 = affine_map<(d0) -> (d0 * 100)>
#map6 = affine_map<(d0)[s0] -> (d0 * 100 + 100, s0 floordiv 4)>
#map7 = affine_map<(d0)[s0] -> (16, d0 * -16 + s0)>
#map8 = affine_map<(d0, d1) -> (d0 * 256 + d1 * 16)>
#map9 = affine_map<(d0) -> (d0 * 8)>
#map10 = affine_map<(d0) -> (d0 * 4)>
#map11 = affine_map<(d0, d1) -> (0)>
#map12 = affine_map<(d0) -> (d0 * 4 + 1)>
#map13 = affine_map<(d0) -> (d0 * 4 + 2)>
#map14 = affine_map<(d0) -> (d0 * 4 + 3)>
#map15 = affine_map<(d0) -> (d0 * 8 + 1)>
#map16 = affine_map<(d0) -> (d0 * 8 + 2)>
#map17 = affine_map<(d0) -> (d0 * 8 + 3)>
#map18 = affine_map<(d0) -> (d0 * 8 + 4)>
#map19 = affine_map<(d0) -> (d0 * 8 + 5)>
#map20 = affine_map<(d0) -> (d0 * 8 + 6)>
#map21 = affine_map<(d0) -> (d0 * 8 + 7)>
module {
  func.func @gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
    %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %2 = memref.dim %arg1, %c1 : memref<?x?xf32>
    affine.for %arg3 = 0 to #map0()[%1] {
      affine.for %arg4 = 0 to #map1()[%0] {
        affine.for %arg5 = 0 to #map2()[%2] {
          affine.for %arg6 = #map3(%arg3) to min #map4(%arg3)[%1] {
            affine.for %arg7 = #map5(%arg4) to min #map6(%arg4)[%0] {
              affine.for %arg8 = 0 to min #map7(%arg5)[%2] {
                %3 = affine.apply #map8(%arg5, %arg8)
                %4 = affine.apply #map9(%arg6)
                %5 = vector.transfer_read %arg1[%4, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %6 = affine.apply #map10(%arg7)
                %7 = vector.transfer_read %arg0[%6, %4], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %8 = vector.transfer_read %arg2[%6, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %9 = arith.mulf %7, %5 : vector<16xf32>
                %10 = arith.addf %8, %9 : vector<16xf32>
                %11 = affine.apply #map12(%arg7)
                %12 = vector.transfer_read %arg0[%11, %4], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %13 = vector.transfer_read %arg2[%11, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %14 = arith.mulf %12, %5 : vector<16xf32>
                %15 = arith.addf %13, %14 : vector<16xf32>
                %16 = affine.apply #map13(%arg7)
                %17 = vector.transfer_read %arg0[%16, %4], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %18 = vector.transfer_read %arg2[%16, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %19 = arith.mulf %17, %5 : vector<16xf32>
                %20 = arith.addf %18, %19 : vector<16xf32>
                %21 = affine.apply #map14(%arg7)
                %22 = vector.transfer_read %arg0[%21, %4], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %23 = vector.transfer_read %arg2[%21, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %24 = arith.mulf %22, %5 : vector<16xf32>
                %25 = arith.addf %23, %24 : vector<16xf32>
                %26 = affine.apply #map15(%arg6)
                %27 = vector.transfer_read %arg1[%26, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %28 = vector.transfer_read %arg0[%6, %26], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %29 = arith.mulf %28, %27 : vector<16xf32>
                %30 = arith.addf %10, %29 : vector<16xf32>
                %31 = vector.transfer_read %arg0[%11, %26], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %32 = arith.mulf %31, %27 : vector<16xf32>
                %33 = arith.addf %15, %32 : vector<16xf32>
                %34 = vector.transfer_read %arg0[%16, %26], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %35 = arith.mulf %34, %27 : vector<16xf32>
                %36 = arith.addf %20, %35 : vector<16xf32>
                %37 = vector.transfer_read %arg0[%21, %26], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %38 = arith.mulf %37, %27 : vector<16xf32>
                %39 = arith.addf %25, %38 : vector<16xf32>
                %40 = affine.apply #map16(%arg6)
                %41 = vector.transfer_read %arg1[%40, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %42 = vector.transfer_read %arg0[%6, %40], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %43 = arith.mulf %42, %41 : vector<16xf32>
                %44 = arith.addf %30, %43 : vector<16xf32>
                %45 = vector.transfer_read %arg0[%11, %40], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %46 = arith.mulf %45, %41 : vector<16xf32>
                %47 = arith.addf %33, %46 : vector<16xf32>
                %48 = vector.transfer_read %arg0[%16, %40], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %49 = arith.mulf %48, %41 : vector<16xf32>
                %50 = arith.addf %36, %49 : vector<16xf32>
                %51 = vector.transfer_read %arg0[%21, %40], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %52 = arith.mulf %51, %41 : vector<16xf32>
                %53 = arith.addf %39, %52 : vector<16xf32>
                %54 = affine.apply #map17(%arg6)
                %55 = vector.transfer_read %arg1[%54, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %56 = vector.transfer_read %arg0[%6, %54], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %57 = arith.mulf %56, %55 : vector<16xf32>
                %58 = arith.addf %44, %57 : vector<16xf32>
                %59 = vector.transfer_read %arg0[%11, %54], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %60 = arith.mulf %59, %55 : vector<16xf32>
                %61 = arith.addf %47, %60 : vector<16xf32>
                %62 = vector.transfer_read %arg0[%16, %54], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %63 = arith.mulf %62, %55 : vector<16xf32>
                %64 = arith.addf %50, %63 : vector<16xf32>
                %65 = vector.transfer_read %arg0[%21, %54], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %66 = arith.mulf %65, %55 : vector<16xf32>
                %67 = arith.addf %53, %66 : vector<16xf32>
                %68 = affine.apply #map18(%arg6)
                %69 = vector.transfer_read %arg1[%68, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %70 = vector.transfer_read %arg0[%6, %68], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %71 = arith.mulf %70, %69 : vector<16xf32>
                %72 = arith.addf %58, %71 : vector<16xf32>
                %73 = vector.transfer_read %arg0[%11, %68], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %74 = arith.mulf %73, %69 : vector<16xf32>
                %75 = arith.addf %61, %74 : vector<16xf32>
                %76 = vector.transfer_read %arg0[%16, %68], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %77 = arith.mulf %76, %69 : vector<16xf32>
                %78 = arith.addf %64, %77 : vector<16xf32>
                %79 = vector.transfer_read %arg0[%21, %68], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %80 = arith.mulf %79, %69 : vector<16xf32>
                %81 = arith.addf %67, %80 : vector<16xf32>
                %82 = affine.apply #map19(%arg6)
                %83 = vector.transfer_read %arg1[%82, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %84 = vector.transfer_read %arg0[%6, %82], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %85 = arith.mulf %84, %83 : vector<16xf32>
                %86 = arith.addf %72, %85 : vector<16xf32>
                %87 = vector.transfer_read %arg0[%11, %82], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %88 = arith.mulf %87, %83 : vector<16xf32>
                %89 = arith.addf %75, %88 : vector<16xf32>
                %90 = vector.transfer_read %arg0[%16, %82], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %91 = arith.mulf %90, %83 : vector<16xf32>
                %92 = arith.addf %78, %91 : vector<16xf32>
                %93 = vector.transfer_read %arg0[%21, %82], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %94 = arith.mulf %93, %83 : vector<16xf32>
                %95 = arith.addf %81, %94 : vector<16xf32>
                %96 = affine.apply #map20(%arg6)
                %97 = vector.transfer_read %arg1[%96, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %98 = vector.transfer_read %arg0[%6, %96], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %99 = arith.mulf %98, %97 : vector<16xf32>
                %100 = arith.addf %86, %99 : vector<16xf32>
                %101 = vector.transfer_read %arg0[%11, %96], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %102 = arith.mulf %101, %97 : vector<16xf32>
                %103 = arith.addf %89, %102 : vector<16xf32>
                %104 = vector.transfer_read %arg0[%16, %96], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %105 = arith.mulf %104, %97 : vector<16xf32>
                %106 = arith.addf %92, %105 : vector<16xf32>
                %107 = vector.transfer_read %arg0[%21, %96], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %108 = arith.mulf %107, %97 : vector<16xf32>
                %109 = arith.addf %95, %108 : vector<16xf32>
                %110 = affine.apply #map21(%arg6)
                %111 = vector.transfer_read %arg1[%110, %3], %cst : memref<?x?xf32>, vector<16xf32>
                %112 = vector.transfer_read %arg0[%6, %110], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %113 = arith.mulf %112, %111 : vector<16xf32>
                %114 = arith.addf %100, %113 : vector<16xf32>
                vector.transfer_write %114, %arg2[%6, %3] : vector<16xf32>, memref<?x?xf32>
                %115 = vector.transfer_read %arg0[%11, %110], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %116 = arith.mulf %115, %111 : vector<16xf32>
                %117 = arith.addf %103, %116 : vector<16xf32>
                vector.transfer_write %117, %arg2[%11, %3] : vector<16xf32>, memref<?x?xf32>
                %118 = vector.transfer_read %arg0[%16, %110], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %119 = arith.mulf %118, %111 : vector<16xf32>
                %120 = arith.addf %106, %119 : vector<16xf32>
                vector.transfer_write %120, %arg2[%16, %3] : vector<16xf32>, memref<?x?xf32>
                %121 = vector.transfer_read %arg0[%21, %110], %cst {permutation_map = #map11} : memref<?x?xf32>, vector<16xf32>
                %122 = arith.mulf %121, %111 : vector<16xf32>
                %123 = arith.addf %109, %122 : vector<16xf32>
                vector.transfer_write %123, %arg2[%21, %3] : vector<16xf32>, memref<?x?xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}

