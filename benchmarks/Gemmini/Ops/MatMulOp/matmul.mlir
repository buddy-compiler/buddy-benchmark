func.func @gemmini_matmul1(%arg0 : memref<16384x64xi8>, %arg1 : memref<64x32xi8>, %arg2 : memref<16384x32xi8>, %arg3 : memref<16384x32xi32>) {
  gemmini.tile_matmul %arg0 %arg1 %arg2 %arg3 : memref<16384x64xi8> memref<64x32xi8> memref<16384x32xi8> memref<16384x32xi32>
  return
}
