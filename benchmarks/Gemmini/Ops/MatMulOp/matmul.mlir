

// CHECK: llvm.func @_mlir_ciface_gemmini_matmul1
func.func @gemmini_matmul1(%arg0 : memref<32x32xi8>, %arg1 : memref<32x32xi8>, %arg2 : memref<32x32xi8>, %arg3 : memref<32x32xi32>) {
  gemmini.tile_matmul %arg0 %arg1 %arg2 %arg3 : memref<32x32xi8> memref<32x32xi8> memref<32x32xi8> memref<32x32xi32>
  return
}

