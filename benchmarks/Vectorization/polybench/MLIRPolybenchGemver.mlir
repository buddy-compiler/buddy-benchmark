
func.func @gemver_init_array(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?x?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %cst_0 = arith.constant 9.000000e+00 : f64
  %cst_1 = arith.constant 8.000000e+00 : f64
  %cst_2 = arith.constant 6.000000e+00 : f64
  %cst_3 = arith.constant 4.000000e+00 : f64
  %cst_4 = arith.constant 2.000000e+00 : f64
  %c1_i32 = arith.constant 1 : i32
  %cst_5 = arith.constant 1.200000e+00 : f64
  %cst_6 = arith.constant 1.500000e+00 : f64
  affine.store %cst_6, %arg1[0] : memref<?xf64>
  affine.store %cst_5, %arg2[0] : memref<?xf64>
  %0 = arith.sitofp %arg0 : i32 to f64
  %1 = arith.index_cast %arg0 : i32 to index
  scf.for %arg12 = %c0 to %1 step %c1 {
    %2 = arith.index_cast %arg12 : index to i32
    %3 = arith.sitofp %2 : i32 to f64
    memref.store %3, %arg4[%arg12] : memref<?xf64>
    %4 = arith.addi %2, %c1_i32 : i32
    %5 = arith.sitofp %4 : i32 to f64
    %6 = arith.divf %5, %0 : f64
    %7 = arith.divf %6, %cst_4 : f64
    memref.store %7, %arg6[%arg12] : memref<?xf64>
    %8 = arith.divf %6, %cst_3 : f64
    memref.store %8, %arg5[%arg12] : memref<?xf64>
    %9 = arith.divf %6, %cst_2 : f64
    memref.store %9, %arg7[%arg12] : memref<?xf64>
    %10 = arith.divf %6, %cst_1 : f64
    memref.store %10, %arg10[%arg12] : memref<?xf64>
    %11 = arith.divf %6, %cst_0 : f64
    memref.store %11, %arg11[%arg12] : memref<?xf64>
    memref.store %cst, %arg9[%arg12] : memref<?xf64>
    memref.store %cst, %arg8[%arg12] : memref<?xf64>
    %12 = arith.index_cast %arg0 : i32 to index
    scf.for %arg13 = %c0 to %12 step %c1 {
      %13 = arith.index_cast %arg13 : index to i32
      %14 = arith.muli %2, %13 : i32
      %15 = arith.remsi %14, %arg0 : i32
      %16 = arith.sitofp %15 : i32 to f64
      %17 = arith.divf %16, %0 : f64
      memref.store %17, %arg3[%arg12, %arg13] : memref<?x?xf64>
    }
  }
  return
}
func.func @gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) {
  %0 = arith.index_cast %arg0 : i32 to index
  affine.for %arg12 = 0 to %0 {
    affine.for %arg13 = 0 to %0 {
      %1 = affine.load %arg3[%arg12, %arg13] : memref<?x?xf64>
      %2 = affine.load %arg4[%arg12] : memref<?xf64>
      %3 = affine.load %arg5[%arg13] : memref<?xf64>
      %4 = arith.mulf %2, %3 : f64
      %5 = arith.addf %1, %4 : f64
      %6 = affine.load %arg6[%arg12] : memref<?xf64>
      %7 = affine.load %arg7[%arg13] : memref<?xf64>
      %8 = arith.mulf %6, %7 : f64
      %9 = arith.addf %5, %8 : f64
      affine.store %9, %arg3[%arg12, %arg13] : memref<?x?xf64>
    }
  }
  affine.for %arg12 = 0 to %0 {
    affine.for %arg13 = 0 to %0 {
      %1 = affine.load %arg9[%arg12] : memref<?xf64>
      %2 = affine.load %arg3[%arg13, %arg12] : memref<?x?xf64>
      %3 = arith.mulf %arg2, %2 : f64
      %4 = affine.load %arg10[%arg13] : memref<?xf64>
      %5 = arith.mulf %3, %4 : f64
      %6 = arith.addf %1, %5 : f64
      affine.store %6, %arg9[%arg12] : memref<?xf64>
    }
  }
  affine.for %arg12 = 0 to %0 {
    %1 = affine.load %arg9[%arg12] : memref<?xf64>
    %2 = affine.load %arg11[%arg12] : memref<?xf64>
    %3 = arith.addf %1, %2 : f64
    affine.store %3, %arg9[%arg12] : memref<?xf64>
  }
  affine.for %arg12 = 0 to %0 {
    affine.for %arg13 = 0 to %0 {
      %1 = affine.load %arg8[%arg12] : memref<?xf64>
      %2 = affine.load %arg3[%arg12, %arg13] : memref<?x?xf64>
      %3 = arith.mulf %arg1, %2 : f64
      %4 = affine.load %arg9[%arg13] : memref<?xf64>
      %5 = arith.mulf %3, %4 : f64
      %6 = arith.addf %1, %5 : f64
      affine.store %6, %arg8[%arg12] : memref<?xf64>
    }
  }
  return
}
