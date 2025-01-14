func.func @mvt_init_array(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4_i32 = arith.constant 4 : i32
  %c3_i32 = arith.constant 3 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg6 = %c0 to %0 step %c1 {
    %1 = arith.index_cast %arg6 : index to i32
    %2 = arith.remsi %1, %arg0 : i32
    %3 = arith.sitofp %2 : i32 to f64
    %4 = arith.sitofp %arg0 : i32 to f64
    %5 = arith.divf %3, %4 : f64
    memref.store %5, %arg1[%arg6] : memref<?xf64>
    %6 = arith.addi %1, %c1_i32 : i32
    %7 = arith.remsi %6, %arg0 : i32
    %8 = arith.sitofp %7 : i32 to f64
    %9 = arith.divf %8, %4 : f64
    memref.store %9, %arg2[%arg6] : memref<?xf64>
    %10 = arith.addi %1, %c3_i32 : i32
    %11 = arith.remsi %10, %arg0 : i32
    %12 = arith.sitofp %11 : i32 to f64
    %13 = arith.divf %12, %4 : f64
    memref.store %13, %arg3[%arg6] : memref<?xf64>
    %14 = arith.addi %1, %c4_i32 : i32
    %15 = arith.remsi %14, %arg0 : i32
    %16 = arith.sitofp %15 : i32 to f64
    %17 = arith.divf %16, %4 : f64
    memref.store %17, %arg4[%arg6] : memref<?xf64>
    %18 = arith.index_cast %arg0 : i32 to index
    scf.for %arg7 = %c0 to %18 step %c1 {
      %19 = arith.index_cast %arg7 : index to i32
      %20 = arith.muli %1, %19 : i32
      %21 = arith.remsi %20, %arg0 : i32
      %22 = arith.sitofp %21 : i32 to f64
      %23 = arith.divf %22, %4 : f64
      memref.store %23, %arg5[%arg6, %arg7] : memref<?x?xf64>
    }
  }
  return
}
func.func @mvt(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?x?xf64>) {
  %0 = arith.index_cast %arg0 : i32 to index
  affine.for %arg6 = 0 to %0 {
    affine.for %arg7 = 0 to %0 {
      %1 = affine.load %arg1[%arg6] : memref<?xf64>
      %2 = affine.load %arg5[%arg6, %arg7] : memref<?x?xf64>
      %3 = affine.load %arg3[%arg7] : memref<?xf64>
      %4 = arith.mulf %2, %3 : f64
      %5 = arith.addf %1, %4 : f64
      affine.store %5, %arg1[%arg6] : memref<?xf64>
    }
  }
  affine.for %arg6 = 0 to %0 {
    affine.for %arg7 = 0 to %0 {
      %1 = affine.load %arg2[%arg6] : memref<?xf64>
      %2 = affine.load %arg5[%arg7, %arg6] : memref<?x?xf64>
      %3 = affine.load %arg4[%arg7] : memref<?xf64>
      %4 = arith.mulf %2, %3 : f64
      %5 = arith.addf %1, %4 : f64
      affine.store %5, %arg2[%arg6] : memref<?xf64>
    }
  }
  return
}
