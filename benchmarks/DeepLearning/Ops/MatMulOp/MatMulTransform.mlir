// This file is generated with transform dialect, see `https://github.com/llvm/llvm-project/tree/6c59f0e1b0fb56c909ad7c9aad4bde37dc006ae0/mlir/test/Dialect/LLVM/transform-e2e.mlir`

#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
module {
  func.func @matmul_transform(%arg0: memref<64x576xf32>, %arg1: memref<576x3136xf32>, %arg2: memref<64x3136xf32>) {
    %cst = arith.constant dense<0.000000e+00> : vector<2x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<8xf32>
    %c4 = arith.constant 4 : index
    %c576 = arith.constant 576 : index
    %c3136 = arith.constant 3136 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c2 = arith.constant 2 : index
    scf.for %arg3 = %c0 to %c64 step %c2 {
      scf.for %arg5 = %c0 to %c3136 step %c64 {
        scf.for %arg7 = %c0 to %c576 step %c4 {
          %3 = vector.load %arg0[%arg3, %arg7] : memref<64x576xf32>, vector<4xf32>
          %4 = affine.apply #map(%arg3)
          %5 = vector.load %arg0[%4, %arg7] : memref<64x576xf32>, vector<4xf32>
          %6 = vector.load %arg1[%arg7, %arg5] : memref<576x3136xf32>, vector<64xf32>
          %7 = affine.apply #map(%arg7)
          %8 = vector.load %arg1[%7, %arg5] : memref<576x3136xf32>, vector<64xf32>
          %9 = affine.apply #map1(%arg7)
          %10 = vector.load %arg1[%9, %arg5] : memref<576x3136xf32>, vector<64xf32>
          %11 = affine.apply #map2(%arg7)
          %12 = vector.load %arg1[%11, %arg5] : memref<576x3136xf32>, vector<64xf32>
          %13 = vector.load %arg2[%arg3, %arg5] : memref<64x3136xf32>, vector<64xf32>
          %14 = vector.insert %13, %cst [0] : vector<64xf32> into vector<2x64xf32>
          %15 = affine.apply #map(%arg3)
          %16 = vector.load %arg2[%15, %arg5] : memref<64x3136xf32>, vector<64xf32>
          %17 = vector.insert %16, %14 [1] : vector<64xf32> into vector<2x64xf32>
          %18 = vector.insert_strided_slice %3, %cst_0 {offsets = [0], strides = [1]} : vector<4xf32> into vector<8xf32>
          %19 = vector.insert_strided_slice %5, %18 {offsets = [4], strides = [1]} : vector<4xf32> into vector<8xf32>
          %20 = vector.shuffle %19, %19 [0, 4, 1, 5, 2, 6, 3, 7] : vector<8xf32>, vector<8xf32>
          %21 = vector.extract_strided_slice %20 {offsets = [0], sizes = [2], strides = [1]} : vector<8xf32> to vector<2xf32>
          %22 = vector.extract_strided_slice %20 {offsets = [2], sizes = [2], strides = [1]} : vector<8xf32> to vector<2xf32>
          %23 = vector.extract_strided_slice %20 {offsets = [4], sizes = [2], strides = [1]} : vector<8xf32> to vector<2xf32>
          %24 = vector.extract_strided_slice %20 {offsets = [6], sizes = [2], strides = [1]} : vector<8xf32> to vector<2xf32>
          %25 = vector.outerproduct %21, %6, %17 {kind = #vector.kind<add>} : vector<2xf32>, vector<64xf32>
          %26 = vector.outerproduct %22, %8, %25 {kind = #vector.kind<add>} : vector<2xf32>, vector<64xf32>
          %27 = vector.outerproduct %23, %10, %26 {kind = #vector.kind<add>} : vector<2xf32>, vector<64xf32>
          %28 = vector.outerproduct %24, %12, %27 {kind = #vector.kind<add>} : vector<2xf32>, vector<64xf32>
          %29 = vector.extract %28[0] : vector<64xf32> from vector<2x64xf32>
          vector.store %29, %arg2[%arg3, %arg5] : memref<64x3136xf32>, vector<64xf32>
          %30 = affine.apply #map(%arg3)
          %31 = vector.extract %28[1] : vector<64xf32> from vector<2x64xf32>
          vector.store %31, %arg2[%30, %arg5] : memref<64x3136xf32>, vector<64xf32>
        }
      }
    }
    return
  }
}
