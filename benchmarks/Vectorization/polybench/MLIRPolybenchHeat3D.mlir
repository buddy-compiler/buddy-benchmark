#map = affine_map<()[s0] -> (s0 - 1)>
#map_0 = affine_map<()[s0] -> (s0 + 1)>

func.func private @heat_3d_init_array(%arg0: i32, %arg1: memref<?x?x?xf64>, %arg2: memref<?x?x?xf64>) attributes {llvm.linkage = #llvm.linkage<internal>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+01 : f64
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg3 = %c0 to %0 step %c1 {
    %1 = arith.index_cast %arg3 : index to i32
    %2 = arith.index_cast %arg0 : i32 to index
    scf.for %arg4 = %c0 to %2 step %c1 {
      %3 = arith.index_cast %arg4 : index to i32
      %4 = arith.index_cast %arg0 : i32 to index
      scf.for %arg5 = %c0 to %4 step %c1 {
        %5 = arith.index_cast %arg5 : index to i32
        %6 = arith.addi %1, %3 : i32
        %7 = arith.subi %arg0, %5 : i32
        %8 = arith.addi %6, %7 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.mulf %9, %cst : f64
        %11 = arith.sitofp %arg0 : i32 to f64
        %12 = arith.divf %10, %11 : f64
        memref.store %12, %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf64>
        %13 = memref.load %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf64>
        memref.store %13, %arg1[%arg3, %arg4, %arg5] : memref<?x?x?xf64>
      }
    }
  }
  return
}

func.func private @heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<?x?x?xf64>, %arg3: memref<?x?x?xf64>) attributes {llvm.linkage = #llvm.linkage<internal>} {
  %cst = arith.constant 1.250000e-01 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg1 : i32 to index
  %2 = arith.index_cast %arg1 : i32 to index
  %3 = arith.index_cast %arg1 : i32 to index
  %4 = arith.index_cast %arg1 : i32 to index
  %5 = arith.index_cast %arg1 : i32 to index
  // manually modified to use parametric loop bounds
  %arg0_cast = arith.index_cast %arg0 : i32 to index
  affine.for %arg4 = 1 to #map_0()[%arg0_cast] {
    affine.for %arg5 = 1 to #map()[%0] {
      affine.for %arg6 = 1 to #map()[%1] {
        affine.for %arg7 = 1 to #map()[%2] {
          %6 = affine.load %arg2[%arg5 + 1, %arg6, %arg7] : memref<?x?x?xf64>
          %7 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<?x?x?xf64>
          %8 = arith.mulf %7, %cst_0 : f64
          %9 = arith.subf %6, %8 : f64
          %10 = affine.load %arg2[%arg5 - 1, %arg6, %arg7] : memref<?x?x?xf64>
          %11 = arith.addf %9, %10 : f64
          %12 = arith.mulf %11, %cst : f64
          %13 = affine.load %arg2[%arg5, %arg6 + 1, %arg7] : memref<?x?x?xf64>
          %14 = arith.subf %13, %8 : f64
          %15 = affine.load %arg2[%arg5, %arg6 - 1, %arg7] : memref<?x?x?xf64>
          %16 = arith.addf %14, %15 : f64
          %17 = arith.mulf %16, %cst : f64
          %18 = arith.addf %12, %17 : f64
          %19 = affine.load %arg2[%arg5, %arg6, %arg7 + 1] : memref<?x?x?xf64>
          %20 = arith.subf %19, %8 : f64
          %21 = affine.load %arg2[%arg5, %arg6, %arg7 - 1] : memref<?x?x?xf64>
          %22 = arith.addf %20, %21 : f64
          %23 = arith.mulf %22, %cst : f64
          %24 = arith.addf %18, %23 : f64
          %25 = arith.addf %24, %7 : f64
          affine.store %25, %arg3[%arg5, %arg6, %arg7] : memref<?x?x?xf64>
        }
      }
    }
    affine.for %arg5 = 1 to #map()[%3] {
      affine.for %arg6 = 1 to #map()[%4] {
        affine.for %arg7 = 1 to #map()[%5] {
          %6 = affine.load %arg3[%arg5 + 1, %arg6, %arg7] : memref<?x?x?xf64>
          %7 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<?x?x?xf64>
          %8 = arith.mulf %7, %cst_0 : f64
          %9 = arith.subf %6, %8 : f64
          %10 = affine.load %arg3[%arg5 - 1, %arg6, %arg7] : memref<?x?x?xf64>
          %11 = arith.addf %9, %10 : f64
          %12 = arith.mulf %11, %cst : f64
          %13 = affine.load %arg3[%arg5, %arg6 + 1, %arg7] : memref<?x?x?xf64>
          %14 = arith.subf %13, %8 : f64
          %15 = affine.load %arg3[%arg5, %arg6 - 1, %arg7] : memref<?x?x?xf64>
          %16 = arith.addf %14, %15 : f64
          %17 = arith.mulf %16, %cst : f64
          %18 = arith.addf %12, %17 : f64
          %19 = affine.load %arg3[%arg5, %arg6, %arg7 + 1] : memref<?x?x?xf64>
          %20 = arith.subf %19, %8 : f64
          %21 = affine.load %arg3[%arg5, %arg6, %arg7 - 1] : memref<?x?x?xf64>
          %22 = arith.addf %20, %21 : f64
          %23 = arith.mulf %22, %cst : f64
          %24 = arith.addf %18, %23 : f64
          %25 = arith.addf %24, %7 : f64
          affine.store %25, %arg2[%arg5, %arg6, %arg7] : memref<?x?x?xf64>
        }
      }
    }
  }
  return
}
