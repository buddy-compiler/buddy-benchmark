#map = affine_map<()[s0] -> (s0 - 1)>

func.func @jacobi_1d_init_array(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 3.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg3 = %c0 to %0 step %c1 {
    %1 = arith.index_cast %arg3 : index to i32
    %2 = arith.sitofp %1 : i32 to f64
    %3 = arith.addf %2, %cst_0 : f64
    %4 = arith.sitofp %arg0 : i32 to f64
    %5 = arith.divf %3, %4 : f64
    memref.store %5, %arg1[%arg3] : memref<?xf64>
    %6 = arith.addf %2, %cst : f64
    %7 = arith.divf %6, %4 : f64
    memref.store %7, %arg2[%arg3] : memref<?xf64>
  }
  return
}

func.func @jacobi_1d(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
  %cst = arith.constant 3.333300e-01 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg1 : i32 to index
  %2 = arith.index_cast %arg0 : i32 to index
  affine.for %arg4 = 0 to %2 {
    affine.for %arg5 = 1 to #map()[%0] {
      %3 = affine.load %arg2[%arg5 - 1] : memref<?xf64>
      %4 = affine.load %arg2[%arg5] : memref<?xf64>
      %5 = arith.addf %3, %4 : f64
      %6 = affine.load %arg2[%arg5 + 1] : memref<?xf64>
      %7 = arith.addf %5, %6 : f64
      %8 = arith.mulf %7, %cst : f64
      affine.store %8, %arg3[%arg5] : memref<?xf64>
    }
    affine.for %arg5 = 1 to #map()[%1] {
      %3 = affine.load %arg3[%arg5 - 1] : memref<?xf64>
      %4 = affine.load %arg3[%arg5] : memref<?xf64>
      %5 = arith.addf %3, %4 : f64
      %6 = affine.load %arg3[%arg5 + 1] : memref<?xf64>
      %7 = arith.addf %5, %6 : f64
      %8 = arith.mulf %7, %cst : f64
      affine.store %8, %arg2[%arg5] : memref<?xf64>
    }
  }
  return
}
