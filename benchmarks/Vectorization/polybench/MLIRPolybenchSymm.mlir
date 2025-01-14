#map = affine_map<(d0) -> (d0)>

func.func @symm_init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant -9.990000e+02 : f64
  %c1_i32 = arith.constant 1 : i32
  %c100_i32 = arith.constant 100 : i32
  %cst_0 = arith.constant 1.200000e+00 : f64
  %cst_1 = arith.constant 1.500000e+00 : f64
  affine.store %cst_1, %arg2[0] : memref<?xf64>
  affine.store %cst_0, %arg3[0] : memref<?xf64>
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg7 = %c0 to %0 step %c1 {
    %2 = arith.index_cast %arg7 : index to i32
    %3 = arith.index_cast %arg1 : i32 to index
    scf.for %arg8 = %c0 to %3 step %c1 {
      %4 = arith.index_cast %arg8 : index to i32
      %5 = arith.addi %2, %4 : i32
      %6 = arith.remsi %5, %c100_i32 : i32
      %7 = arith.sitofp %6 : i32 to f64
      %8 = arith.sitofp %arg0 : i32 to f64
      %9 = arith.divf %7, %8 : f64
      memref.store %9, %arg4[%arg7, %arg8] : memref<?x?xf64>
      %10 = arith.addi %arg1, %2 : i32
      %11 = arith.subi %10, %4 : i32
      %12 = arith.remsi %11, %c100_i32 : i32
      %13 = arith.sitofp %12 : i32 to f64
      %14 = arith.divf %13, %8 : f64
      memref.store %14, %arg6[%arg7, %arg8] : memref<?x?xf64>
    }
  }
  %1 = arith.index_cast %arg0 : i32 to index
  scf.for %arg7 = %c0 to %1 step %c1 {
    %2 = arith.index_cast %arg7 : index to i32
    %3 = arith.addi %2, %c1_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    scf.for %arg8 = %c0 to %4 step %c1 {
      %8 = arith.index_cast %arg8 : index to i32
      %9 = arith.addi %2, %8 : i32
      %10 = arith.remsi %9, %c100_i32 : i32
      %11 = arith.sitofp %10 : i32 to f64
      %12 = arith.sitofp %arg0 : i32 to f64
      %13 = arith.divf %11, %12 : f64
      memref.store %13, %arg5[%arg7, %arg8] : memref<?x?xf64>
    }
    %5 = arith.addi %2, %c1_i32 : i32
    %6 = arith.index_cast %arg0 : i32 to index
    %7 = arith.index_cast %5 : i32 to index
    scf.for %arg8 = %7 to %6 step %c1 {
      %8 = arith.subi %arg8, %7 : index
      %9 = arith.index_cast %5 : i32 to index
      %10 = arith.addi %9, %8 : index
      memref.store %cst, %arg5[%arg7, %10] : memref<?x?xf64>
    }
  }
  return
}

func.func @symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  %alloca = memref.alloca() : memref<f64>
  %1 = llvm.mlir.undef : f64
  affine.store %1, %alloca[] : memref<f64>
  %2 = arith.index_cast %arg0 : i32 to index
  affine.for %arg7 = 0 to %2 {
    affine.for %arg8 = 0 to %0 {
      affine.store %cst, %alloca[] : memref<f64>
      affine.for %arg9 = 0 to #map(%arg7) {
        %13 = affine.load %arg6[%arg7, %arg8] : memref<?x?xf64>
        %14 = arith.mulf %arg2, %13 : f64
        %15 = affine.load %arg5[%arg7, %arg9] : memref<?x?xf64>
        %16 = arith.mulf %14, %15 : f64
        %17 = affine.load %arg4[%arg9, %arg8] : memref<?x?xf64>
        %18 = arith.addf %17, %16 : f64
        affine.store %18, %arg4[%arg9, %arg8] : memref<?x?xf64>
        %19 = affine.load %arg6[%arg9, %arg8] : memref<?x?xf64>
        %20 = affine.load %arg5[%arg7, %arg9] : memref<?x?xf64>
        %21 = arith.mulf %19, %20 : f64
        %22 = affine.load %alloca[] : memref<f64>
        %23 = arith.addf %22, %21 : f64
        affine.store %23, %alloca[] : memref<f64>
      }
      %3 = affine.load %arg4[%arg7, %arg8] : memref<?x?xf64>
      %4 = arith.mulf %arg3, %3 : f64
      %5 = affine.load %arg6[%arg7, %arg8] : memref<?x?xf64>
      %6 = arith.mulf %arg2, %5 : f64
      %7 = affine.load %arg5[%arg7, %arg7] : memref<?x?xf64>
      %8 = arith.mulf %6, %7 : f64
      %9 = arith.addf %4, %8 : f64
      %10 = affine.load %alloca[] : memref<f64>
      %11 = arith.mulf %arg2, %10 : f64
      %12 = arith.addf %9, %11 : f64
      affine.store %12, %arg4[%arg7, %arg8] : memref<?x?xf64>
    }
  }
  return
}
