// RUN: mlir-opt %s --convert-linalg-to-loops --convert-scf-to-cf --lower-affine --finalize-memref-to-llvm --llvm-request-c-wrappers --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | llc -O3 -filetype=obj -o %t.o

func.func @batch_matmul(%arg0: memref<?x?x?xi32>, %arg1: memref<?x?x?xi32>, %arg2: memref<?x?x?xi32>) {
  linalg.batch_matmul
    ins(%arg0, %arg1: memref<?x?x?xi32>, memref<?x?x?xi32>)
    outs(%arg2: memref<?x?x?xi32>)
  return
}