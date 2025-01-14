#map = affine_map<(d0) -> (d0 + 1)>

func.func @trmm_init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 1.500000e+00 : f64
  affine.store %cst_0, %arg2[0] : memref<?xf64>
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg5 = %c0 to %0 step %c1 {
    %1 = arith.index_cast %arg5 : index to i32
    scf.for %arg6 = %c0 to %arg5 step %c1 {
      %3 = arith.index_cast %arg6 : index to i32
      %4 = arith.addi %1, %3 : i32
      %5 = arith.remsi %4, %arg0 : i32
      %6 = arith.sitofp %5 : i32 to f64
      %7 = arith.sitofp %arg0 : i32 to f64
      %8 = arith.divf %6, %7 : f64
      memref.store %8, %arg3[%arg5, %arg6] : memref<?x?xf64>
    }
    memref.store %cst, %arg3[%arg5, %arg5] : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    scf.for %arg6 = %c0 to %2 step %c1 {
      %3 = arith.index_cast %arg6 : index to i32
      %4 = arith.subi %1, %3 : i32
      %5 = arith.addi %arg1, %4 : i32
      %6 = arith.remsi %5, %arg1 : i32
      %7 = arith.sitofp %6 : i32 to f64
      %8 = arith.sitofp %arg1 : i32 to f64
      %9 = arith.divf %7, %8 : f64
      memref.store %9, %arg4[%arg5, %arg6] : memref<?x?xf64>
    }
  }
  return
}

func.func @trmm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>) {
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg0 : i32 to index
  affine.for %arg5 = 0 to %1 {
    affine.for %arg6 = 0 to %0 {
      affine.for %arg7 = #map(%arg5) to %1 {
        %4 = affine.load %arg3[%arg7, %arg5] : memref<?x?xf64>
        %5 = affine.load %arg4[%arg7, %arg6] : memref<?x?xf64>
        %6 = arith.mulf %4, %5 : f64
        %7 = affine.load %arg4[%arg5, %arg6] : memref<?x?xf64>
        %8 = arith.addf %7, %6 : f64
        affine.store %8, %arg4[%arg5, %arg6] : memref<?x?xf64>
      }
      %2 = affine.load %arg4[%arg5, %arg6] : memref<?x?xf64>
      %3 = arith.mulf %arg2, %2 : f64
      affine.store %3, %arg4[%arg5, %arg6] : memref<?x?xf64>
    }
  }
  return
}
