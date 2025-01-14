#map = affine_map<()[s0] -> (s0 - 1)>
func.func @fdtd_2d_init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3_i32 = arith.constant 3 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg7 = %c0 to %0 step %c1 {
    %2 = arith.index_cast %arg7 : index to i32
    %3 = arith.sitofp %2 : i32 to f64
    memref.store %3, %arg6[%arg7] : memref<?xf64>
  }
  %1 = arith.index_cast %arg1 : i32 to index
  scf.for %arg7 = %c0 to %1 step %c1 {
    %2 = arith.index_cast %arg7 : index to i32
    %3 = arith.index_cast %arg2 : i32 to index
    scf.for %arg8 = %c0 to %3 step %c1 {
      %4 = arith.index_cast %arg8 : index to i32
      %5 = arith.sitofp %2 : i32 to f64
      %6 = arith.addi %4, %c1_i32 : i32
      %7 = arith.sitofp %6 : i32 to f64
      %8 = arith.mulf %5, %7 : f64
      %9 = arith.sitofp %arg1 : i32 to f64
      %10 = arith.divf %8, %9 : f64
      memref.store %10, %arg3[%arg7, %arg8] : memref<?x?xf64>
      %11 = arith.addi %4, %c2_i32 : i32
      %12 = arith.sitofp %11 : i32 to f64
      %13 = arith.mulf %5, %12 : f64
      %14 = arith.sitofp %arg2 : i32 to f64
      %15 = arith.divf %13, %14 : f64
      memref.store %15, %arg4[%arg7, %arg8] : memref<?x?xf64>
      %16 = arith.addi %4, %c3_i32 : i32
      %17 = arith.sitofp %16 : i32 to f64
      %18 = arith.mulf %5, %17 : f64
      %19 = arith.divf %18, %9 : f64
      memref.store %19, %arg5[%arg7, %arg8] : memref<?x?xf64>
    }
  }
  return
}

func.func @fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?xf64>) {
  %cst = arith.constant 5.000000e-01 : f64
  %cst_0 = arith.constant 0.69999999999999996 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg2 : i32 to index
  %2 = arith.index_cast %arg0 : i32 to index
  affine.for %arg7 = 0 to %2 {
    affine.for %arg8 = 0 to %1 {
      %3 = affine.load %arg6[%arg7] : memref<?xf64>
      affine.store %3, %arg4[0, %arg8] : memref<?x?xf64>
    }
    affine.for %arg8 = 1 to %0 {
      affine.for %arg9 = 0 to %1 {
        %3 = affine.load %arg4[%arg8, %arg9] : memref<?x?xf64>
        %4 = affine.load %arg5[%arg8, %arg9] : memref<?x?xf64>
        %5 = affine.load %arg5[%arg8 - 1, %arg9] : memref<?x?xf64>
        %6 = arith.subf %4, %5 : f64
        %7 = arith.mulf %6, %cst : f64
        %8 = arith.subf %3, %7 : f64
        affine.store %8, %arg4[%arg8, %arg9] : memref<?x?xf64>
      }
    }
    affine.for %arg8 = 0 to %0 {
      affine.for %arg9 = 1 to %1 {
        %3 = affine.load %arg3[%arg8, %arg9] : memref<?x?xf64>
        %4 = affine.load %arg5[%arg8, %arg9] : memref<?x?xf64>
        %5 = affine.load %arg5[%arg8, %arg9 - 1] : memref<?x?xf64>
        %6 = arith.subf %4, %5 : f64
        %7 = arith.mulf %6, %cst : f64
        %8 = arith.subf %3, %7 : f64
        affine.store %8, %arg3[%arg8, %arg9] : memref<?x?xf64>
      }
    }
    affine.for %arg8 = 0 to #map()[%0] {
      affine.for %arg9 = 0 to #map()[%1] {
        %3 = affine.load %arg5[%arg8, %arg9] : memref<?x?xf64>
        %4 = affine.load %arg3[%arg8, %arg9 + 1] : memref<?x?xf64>
        %5 = affine.load %arg3[%arg8, %arg9] : memref<?x?xf64>
        %6 = arith.subf %4, %5 : f64
        %7 = affine.load %arg4[%arg8 + 1, %arg9] : memref<?x?xf64>
        %8 = arith.addf %6, %7 : f64
        %9 = affine.load %arg4[%arg8, %arg9] : memref<?x?xf64>
        %10 = arith.subf %8, %9 : f64
        %11 = arith.mulf %10, %cst_0 : f64
        %12 = arith.subf %3, %11 : f64
        affine.store %12, %arg5[%arg8, %arg9] : memref<?x?xf64>
      }
    }
  }
  return
}
