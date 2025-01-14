#map = affine_map<(d0) -> (d0 + 1)>

func.func @syr2k_init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3_i32 = arith.constant 3 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.200000e+00 : f64
  %cst_0 = arith.constant 1.500000e+00 : f64
  affine.store %cst_0, %arg2[0] : memref<?xf64>
  affine.store %cst, %arg3[0] : memref<?xf64>
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg7 = %c0 to %0 step %c1 {
    %2 = arith.index_cast %arg7 : index to i32
    %3 = arith.index_cast %arg1 : i32 to index
    scf.for %arg8 = %c0 to %3 step %c1 {
      %4 = arith.index_cast %arg8 : index to i32
      %5 = arith.muli %2, %4 : i32
      %6 = arith.addi %5, %c1_i32 : i32
      %7 = arith.remsi %6, %arg0 : i32
      %8 = arith.sitofp %7 : i32 to f64
      %9 = arith.sitofp %arg0 : i32 to f64
      %10 = arith.divf %8, %9 : f64
      memref.store %10, %arg5[%arg7, %arg8] : memref<?x?xf64>
      %11 = arith.addi %5, %c2_i32 : i32
      %12 = arith.remsi %11, %arg1 : i32
      %13 = arith.sitofp %12 : i32 to f64
      %14 = arith.sitofp %arg1 : i32 to f64
      %15 = arith.divf %13, %14 : f64
      memref.store %15, %arg6[%arg7, %arg8] : memref<?x?xf64>
    }
  }
  %1 = arith.index_cast %arg0 : i32 to index
  scf.for %arg7 = %c0 to %1 step %c1 {
    %2 = arith.index_cast %arg7 : index to i32
    %3 = arith.index_cast %arg0 : i32 to index
    scf.for %arg8 = %c0 to %3 step %c1 {
      %4 = arith.index_cast %arg8 : index to i32
      %5 = arith.muli %2, %4 : i32
      %6 = arith.addi %5, %c3_i32 : i32
      %7 = arith.remsi %6, %arg0 : i32
      %8 = arith.sitofp %7 : i32 to f64
      %9 = arith.sitofp %arg1 : i32 to f64
      %10 = arith.divf %8, %9 : f64
      memref.store %10, %arg4[%arg7, %arg8] : memref<?x?xf64>
    }
  }
  return
}

func.func @syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) {
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg0 : i32 to index
  affine.for %arg7 = 0 to %1 {
    affine.for %arg8 = 0 to #map(%arg7) {
      %2 = affine.load %arg4[%arg7, %arg8] : memref<?x?xf64>
      %3 = arith.mulf %2, %arg3 : f64
      affine.store %3, %arg4[%arg7, %arg8] : memref<?x?xf64>
    }
    affine.for %arg8 = 0 to %0 {
      affine.for %arg9 = 0 to #map(%arg7) {
        %2 = affine.load %arg5[%arg9, %arg8] : memref<?x?xf64>
        %3 = arith.mulf %2, %arg2 : f64
        %4 = affine.load %arg6[%arg7, %arg8] : memref<?x?xf64>
        %5 = arith.mulf %3, %4 : f64
        %6 = affine.load %arg6[%arg9, %arg8] : memref<?x?xf64>
        %7 = arith.mulf %6, %arg2 : f64
        %8 = affine.load %arg5[%arg7, %arg8] : memref<?x?xf64>
        %9 = arith.mulf %7, %8 : f64
        %10 = arith.addf %5, %9 : f64
        %11 = affine.load %arg4[%arg7, %arg9] : memref<?x?xf64>
        %12 = arith.addf %11, %10 : f64
        affine.store %12, %arg4[%arg7, %arg9] : memref<?x?xf64>
      }
    }
  }
  return
}
