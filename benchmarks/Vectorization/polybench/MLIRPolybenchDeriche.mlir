func.func @deriche_init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 6.553500e+04 : f32
  %c65536_i32 = arith.constant 65536 : i32
  %c991_i32 = arith.constant 991 : i32
  %c313_i32 = arith.constant 313 : i32
  %cst_0 = arith.constant 2.500000e-01 : f32
  affine.store %cst_0, %arg2[0] : memref<?xf32>
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg5 = %c0 to %0 step %c1 {
    %1 = arith.index_cast %arg5 : index to i32
    %2 = arith.index_cast %arg1 : i32 to index
    scf.for %arg6 = %c0 to %2 step %c1 {
      %3 = arith.index_cast %arg6 : index to i32
      %4 = arith.muli %1, %c313_i32 : i32
      %5 = arith.muli %3, %c991_i32 : i32
      %6 = arith.addi %4, %5 : i32
      %7 = arith.remsi %6, %c65536_i32 : i32
      %8 = arith.sitofp %7 : i32 to f32
      %9 = arith.divf %8, %cst : f32
      memref.store %9, %arg3[%arg5, %arg6] : memref<?x?xf32>
    }
  }
  return
}
func.func @deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: memref<?x?xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %cst_1 = arith.constant -2.000000e+00 : f32
  %cst_2 = arith.constant 0.000000e+00 : f32
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg1 : i32 to index
  %2 = arith.index_cast %arg1 : i32 to index
  %3 = arith.index_cast %arg1 : i32 to index
  %4 = arith.index_cast %arg1 : i32 to index
  %5 = arith.index_cast %arg0 : i32 to index
  %6 = arith.index_cast %arg0 : i32 to index
  %7 = arith.index_cast %arg0 : i32 to index
  %8 = arith.index_cast %arg0 : i32 to index
  %9 = llvm.mlir.undef : f32
  %alloca = memref.alloca() : memref<f32>
  affine.store %9, %alloca[] : memref<f32>
  %alloca_3 = memref.alloca() : memref<f32>
  affine.store %9, %alloca_3[] : memref<f32>
  %alloca_4 = memref.alloca() : memref<f32>
  affine.store %9, %alloca_4[] : memref<f32>
  %alloca_5 = memref.alloca() : memref<f32>
  affine.store %9, %alloca_5[] : memref<f32>
  %alloca_6 = memref.alloca() : memref<f32>
  affine.store %9, %alloca_6[] : memref<f32>
  %alloca_7 = memref.alloca() : memref<f32>
  affine.store %9, %alloca_7[] : memref<f32>
  %alloca_8 = memref.alloca() : memref<f32>
  affine.store %9, %alloca_8[] : memref<f32>
  %alloca_9 = memref.alloca() : memref<f32>
  affine.store %9, %alloca_9[] : memref<f32>
  %alloca_10 = memref.alloca() : memref<f32>
  affine.store %9, %alloca_10[] : memref<f32>
  %alloca_11 = memref.alloca() : memref<f32>
  affine.store %9, %alloca_11[] : memref<f32>
  %10 = arith.negf %arg2 : f32
  %11 = math.exp %10 : f32
  %12 = arith.subf %cst, %11 : f32
  %13 = arith.mulf %12, %12 : f32
  %14 = arith.mulf %arg2, %cst_0 : f32
  %15 = arith.mulf %14, %11 : f32
  %16 = arith.addf %15, %cst : f32
  %17 = math.exp %14 : f32
  %18 = arith.subf %16, %17 : f32
  %19 = arith.divf %13, %18 : f32
  %20 = arith.mulf %19, %11 : f32
  %21 = arith.subf %arg2, %cst : f32
  %22 = arith.mulf %20, %21 : f32
  %23 = arith.addf %arg2, %cst : f32
  %24 = arith.mulf %20, %23 : f32
  %25 = arith.negf %19 : f32
  %26 = arith.mulf %arg2, %cst_1 : f32
  %27 = math.exp %26 : f32
  %28 = arith.mulf %25, %27 : f32
  %29 = math.powf %cst_0, %10 : f32
  %30 = arith.negf %27 : f32
  %31 = arith.index_cast %arg0 : i32 to index
  affine.for %arg7 = 0 to %31 {
    affine.store %cst_2, %alloca_9[] : memref<f32>
    affine.store %cst_2, %alloca_8[] : memref<f32>
    affine.store %cst_2, %alloca_11[] : memref<f32>
    affine.for %arg8 = 0 to %0 {
      %33 = affine.load %arg3[%arg7, %arg8] : memref<?x?xf32>
      %34 = arith.mulf %19, %33 : f32
      %35 = affine.load %alloca_11[] : memref<f32>
      %36 = arith.mulf %22, %35 : f32
      %37 = arith.addf %34, %36 : f32
      %38 = affine.load %alloca_9[] : memref<f32>
      %39 = arith.mulf %29, %38 : f32
      %40 = arith.addf %37, %39 : f32
      %41 = affine.load %alloca_8[] : memref<f32>
      %42 = arith.mulf %30, %41 : f32
      %43 = arith.addf %40, %42 : f32
      affine.store %43, %arg5[%arg7, %arg8] : memref<?x?xf32>
      %44 = affine.load %arg3[%arg7, %arg8] : memref<?x?xf32>
      affine.store %44, %alloca_11[] : memref<f32>
      affine.store %38, %alloca_8[] : memref<f32>
      %45 = affine.load %arg5[%arg7, %arg8] : memref<?x?xf32>
      affine.store %45, %alloca_9[] : memref<f32>
    }
  }
  affine.for %arg7 = 0 to %31 {
    affine.store %cst_2, %alloca_3[] : memref<f32>
    affine.store %cst_2, %alloca[] : memref<f32>
    affine.store %cst_2, %alloca_7[] : memref<f32>
    affine.store %cst_2, %alloca_6[] : memref<f32>
    affine.for %arg8 = 0 to %0 {
      %33 = affine.load %alloca_7[] : memref<f32>
      %34 = arith.mulf %24, %33 : f32
      %35 = affine.load %alloca_6[] : memref<f32>
      %36 = arith.mulf %28, %35 : f32
      %37 = arith.addf %34, %36 : f32
      %38 = affine.load %alloca_3[] : memref<f32>
      %39 = arith.mulf %29, %38 : f32
      %40 = arith.addf %37, %39 : f32
      %41 = affine.load %alloca[] : memref<f32>
      %42 = arith.mulf %30, %41 : f32
      %43 = arith.addf %40, %42 : f32
      affine.store %43, %arg6[%arg7, -%arg8 + symbol(%1) - 1] : memref<?x?xf32>
      affine.store %33, %alloca_6[] : memref<f32>
      %44 = affine.load %arg3[%arg7, -%arg8 + symbol(%2) - 1] : memref<?x?xf32>
      affine.store %44, %alloca_7[] : memref<f32>
      affine.store %38, %alloca[] : memref<f32>
      %45 = affine.load %arg6[%arg7, -%arg8 + symbol(%3) - 1] : memref<?x?xf32>
      affine.store %45, %alloca_3[] : memref<f32>
    }
  }
  affine.for %arg7 = 0 to %31 {
    affine.for %arg8 = 0 to %4 {
      %33 = affine.load %arg5[%arg7, %arg8] : memref<?x?xf32>
      %34 = affine.load %arg6[%arg7, %arg8] : memref<?x?xf32>
      %35 = arith.addf %33, %34 : f32
      affine.store %35, %arg4[%arg7, %arg8] : memref<?x?xf32>
    }
  }
  %32 = arith.index_cast %arg1 : i32 to index
  affine.for %arg7 = 0 to %32 {
    affine.store %cst_2, %alloca_10[] : memref<f32>
    affine.store %cst_2, %alloca_9[] : memref<f32>
    affine.store %cst_2, %alloca_8[] : memref<f32>
    affine.for %arg8 = 0 to %31 {
      %33 = affine.load %arg4[%arg8, %arg7] : memref<?x?xf32>
      %34 = arith.mulf %19, %33 : f32
      %35 = affine.load %alloca_10[] : memref<f32>
      %36 = arith.mulf %22, %35 : f32
      %37 = arith.addf %34, %36 : f32
      %38 = affine.load %alloca_9[] : memref<f32>
      %39 = arith.mulf %29, %38 : f32
      %40 = arith.addf %37, %39 : f32
      %41 = affine.load %alloca_8[] : memref<f32>
      %42 = arith.mulf %30, %41 : f32
      %43 = arith.addf %40, %42 : f32
      affine.store %43, %arg5[%arg8, %arg7] : memref<?x?xf32>
      %44 = affine.load %arg4[%arg8, %arg7] : memref<?x?xf32>
      affine.store %44, %alloca_10[] : memref<f32>
      affine.store %38, %alloca_8[] : memref<f32>
      %45 = affine.load %arg5[%arg8, %arg7] : memref<?x?xf32>
      affine.store %45, %alloca_9[] : memref<f32>
    }
  }
  affine.for %arg7 = 0 to %32 {
    affine.store %cst_2, %alloca_5[] : memref<f32>
    affine.store %cst_2, %alloca_4[] : memref<f32>
    affine.store %cst_2, %alloca_3[] : memref<f32>
    affine.store %cst_2, %alloca[] : memref<f32>
    affine.for %arg8 = 0 to %5 {
      %33 = affine.load %alloca_5[] : memref<f32>
      %34 = arith.mulf %24, %33 : f32
      %35 = affine.load %alloca_4[] : memref<f32>
      %36 = arith.mulf %28, %35 : f32
      %37 = arith.addf %34, %36 : f32
      %38 = affine.load %alloca_3[] : memref<f32>
      %39 = arith.mulf %29, %38 : f32
      %40 = arith.addf %37, %39 : f32
      %41 = affine.load %alloca[] : memref<f32>
      %42 = arith.mulf %30, %41 : f32
      %43 = arith.addf %40, %42 : f32
      affine.store %43, %arg6[-%arg8 + symbol(%6) - 1, %arg7] : memref<?x?xf32>
      affine.store %33, %alloca_4[] : memref<f32>
      %44 = affine.load %arg4[-%arg8 + symbol(%7) - 1, %arg7] : memref<?x?xf32>
      affine.store %44, %alloca_5[] : memref<f32>
      affine.store %38, %alloca[] : memref<f32>
      %45 = affine.load %arg6[-%arg8 + symbol(%8) - 1, %arg7] : memref<?x?xf32>
      affine.store %45, %alloca_3[] : memref<f32>
    }
  }
  affine.for %arg7 = 0 to %31 {
    affine.for %arg8 = 0 to %32 {
      %33 = affine.load %arg5[%arg7, %arg8] : memref<?x?xf32>
      %34 = affine.load %arg6[%arg7, %arg8] : memref<?x?xf32>
      %35 = arith.addf %33, %34 : f32
      affine.store %35, %arg4[%arg7, %arg8] : memref<?x?xf32>
    }
  }
  return
}
