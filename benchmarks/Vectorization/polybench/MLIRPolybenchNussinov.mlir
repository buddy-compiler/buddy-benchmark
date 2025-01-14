#map = affine_map<(d0)[s0] -> (-d0 + s0)>
#map1 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 - 1 >= 0, d1 - 1 >= 0)>
#set2 = affine_set<(d0, d1)[s0] : (d0 + d1 - s0 - 1 >= 0)>

func.func @nussinov_init_array(%arg0: i32, %arg1: memref<?xi8>, %arg2: memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg3 = %c0 to %0 step %c1 {
    %2 = arith.index_cast %arg3 : index to i32
    %3 = arith.addi %2, %c1_i32 : i32
    %4 = arith.remsi %3, %c4_i32 : i32
    %5 = arith.trunci %4 : i32 to i8
    memref.store %5, %arg1[%arg3] : memref<?xi8>
  }
  %1 = arith.index_cast %arg0 : i32 to index
  scf.for %arg3 = %c0 to %1 step %c1 {
    %2 = arith.index_cast %arg0 : i32 to index
    scf.for %arg4 = %c0 to %2 step %c1 {
      memref.store %c0_i32, %arg2[%arg3, %arg4] : memref<?x?xi32>
    }
  }
  return
}

func.func @nussinov(%arg0: i32, %arg1: memref<?xi8>, %arg2: memref<?x?xi32>) {
  %c3_i32 = arith.constant 3 : i32
  %0 = arith.index_cast %arg0 : i32 to index
  %1 = arith.index_cast %arg0 : i32 to index
  %2 = arith.index_cast %arg0 : i32 to index
  %3 = arith.index_cast %arg0 : i32 to index
  %4 = arith.index_cast %arg0 : i32 to index
  %5 = arith.index_cast %arg0 : i32 to index
  %6 = arith.index_cast %arg0 : i32 to index
  %7 = arith.index_cast %arg0 : i32 to index
  %8 = arith.index_cast %arg0 : i32 to index
  %9 = arith.index_cast %arg0 : i32 to index
  %10 = arith.index_cast %arg0 : i32 to index
  %11 = arith.index_cast %arg0 : i32 to index
  %12 = arith.index_cast %arg0 : i32 to index
  %13 = arith.index_cast %arg0 : i32 to index
  %14 = arith.index_cast %arg0 : i32 to index
  %15 = arith.index_cast %arg0 : i32 to index
  %16 = arith.index_cast %arg0 : i32 to index
  %17 = arith.index_cast %arg0 : i32 to index
  %18 = arith.index_cast %arg0 : i32 to index
  %19 = arith.index_cast %arg0 : i32 to index
  %20 = arith.index_cast %arg0 : i32 to index
  %21 = arith.index_cast %arg0 : i32 to index
  %22 = arith.index_cast %arg0 : i32 to index
  %23 = arith.index_cast %arg0 : i32 to index
  %24 = arith.index_cast %arg0 : i32 to index
  %25 = arith.index_cast %arg0 : i32 to index
  %26 = arith.index_cast %arg0 : i32 to index
  %27 = arith.index_cast %arg0 : i32 to index
  %28 = arith.index_cast %arg0 : i32 to index
  %29 = arith.index_cast %arg0 : i32 to index
  %30 = arith.index_cast %arg0 : i32 to index
  affine.for %arg3 = 0 to %0 {
    affine.for %arg4 = #map(%arg3)[%1] to %1 {
      affine.if #set(%arg4) {
        %31 = affine.load %arg2[-%arg3 + symbol(%2) - 1, %arg4] : memref<?x?xi32>
        %32 = affine.load %arg2[-%arg3 + symbol(%3) - 1, %arg4 - 1] : memref<?x?xi32>
        %33 = arith.cmpi sge, %31, %32 : i32
        %34 = scf.if %33 -> (i32) {
          %35 = affine.load %arg2[-%arg3 + symbol(%4) - 1, %arg4] : memref<?x?xi32>
          scf.yield %35 : i32
        } else {
          %35 = affine.load %arg2[-%arg3 + symbol(%5) - 1, %arg4 - 1] : memref<?x?xi32>
          scf.yield %35 : i32
        }
        affine.store %34, %arg2[-%arg3 + symbol(%6) - 1, %arg4] : memref<?x?xi32>
      }
      affine.if #set(%arg3) {
        %31 = affine.load %arg2[-%arg3 + symbol(%7) - 1, %arg4] : memref<?x?xi32>
        %32 = affine.load %arg2[-%arg3 + symbol(%8), %arg4] : memref<?x?xi32>
        %33 = arith.cmpi sge, %31, %32 : i32
        %34 = scf.if %33 -> (i32) {
          %35 = affine.load %arg2[-%arg3 + symbol(%9) - 1, %arg4] : memref<?x?xi32>
          scf.yield %35 : i32
        } else {
          %35 = affine.load %arg2[-%arg3 + symbol(%10), %arg4] : memref<?x?xi32>
          scf.yield %35 : i32
        }
        affine.store %34, %arg2[-%arg3 + symbol(%11) - 1, %arg4] : memref<?x?xi32>
      }
      affine.if #set1(%arg4, %arg3) {
        affine.if #set2(%arg3, %arg4)[%13] {
          %31 = affine.load %arg2[-%arg3 + symbol(%12) - 1, %arg4] : memref<?x?xi32>
          %32 = affine.load %arg2[-%arg3 + symbol(%14), %arg4 - 1] : memref<?x?xi32>
          %33 = affine.load %arg1[-%arg3 + symbol(%15) - 1] : memref<?xi8>
          %34 = arith.extsi %33 : i8 to i32
          %35 = affine.load %arg1[%arg4] : memref<?xi8>
          %36 = arith.extsi %35 : i8 to i32
          %37 = arith.addi %34, %36 : i32
          %38 = arith.cmpi eq, %37, %c3_i32 : i32
          %39 = arith.extui %38 : i1 to i32
          %40 = arith.addi %32, %39 : i32
          %41 = arith.cmpi sge, %31, %40 : i32
          %42 = scf.if %41 -> (i32) {
            %43 = affine.load %arg2[-%arg3 + symbol(%16) - 1, %arg4] : memref<?x?xi32>
            scf.yield %43 : i32
          } else {
            %43 = affine.load %arg2[-%arg3 + symbol(%17), %arg4 - 1] : memref<?x?xi32>
            %44 = affine.load %arg1[-%arg3 + symbol(%18) - 1] : memref<?xi8>
            %45 = arith.extsi %44 : i8 to i32
            %46 = arith.extsi %35 : i8 to i32
            %47 = arith.addi %45, %46 : i32
            %48 = arith.cmpi eq, %47, %c3_i32 : i32
            %49 = arith.extui %48 : i1 to i32
            %50 = arith.addi %43, %49 : i32
            scf.yield %50 : i32
          }
          affine.store %42, %arg2[-%arg3 + symbol(%19) - 1, %arg4] : memref<?x?xi32>
        } else {
          %31 = affine.load %arg2[-%arg3 + symbol(%20) - 1, %arg4] : memref<?x?xi32>
          %32 = affine.load %arg2[-%arg3 + symbol(%21), %arg4 - 1] : memref<?x?xi32>
          %33 = arith.cmpi sge, %31, %32 : i32
          %34 = scf.if %33 -> (i32) {
            %35 = affine.load %arg2[-%arg3 + symbol(%22) - 1, %arg4] : memref<?x?xi32>
            scf.yield %35 : i32
          } else {
            %35 = affine.load %arg2[-%arg3 + symbol(%23), %arg4 - 1] : memref<?x?xi32>
            scf.yield %35 : i32
          }
          affine.store %34, %arg2[-%arg3 + symbol(%24) - 1, %arg4] : memref<?x?xi32>
        }
      }
      affine.for %arg5 = #map(%arg3)[%25] to #map1(%arg4) {
        %31 = affine.load %arg2[-%arg3 + symbol(%26) - 1, %arg4] : memref<?x?xi32>
        %32 = affine.load %arg2[-%arg3 + symbol(%27) - 1, %arg5] : memref<?x?xi32>
        %33 = affine.load %arg2[%arg5 + 1, %arg4] : memref<?x?xi32>
        %34 = arith.addi %32, %33 : i32
        %35 = arith.cmpi sge, %31, %34 : i32
        %36 = scf.if %35 -> (i32) {
          %37 = affine.load %arg2[-%arg3 + symbol(%28) - 1, %arg4] : memref<?x?xi32>
          scf.yield %37 : i32
        } else {
          %37 = affine.load %arg2[-%arg3 + symbol(%29) - 1, %arg5] : memref<?x?xi32>
          %38 = arith.addi %37, %33 : i32
          scf.yield %38 : i32
        }
        affine.store %36, %arg2[-%arg3 + symbol(%30) - 1, %arg4] : memref<?x?xi32>
      }
    }
  }
  return
}
