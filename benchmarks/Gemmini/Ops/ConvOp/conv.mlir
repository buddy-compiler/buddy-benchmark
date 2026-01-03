
func.func @gemmini_conv_3(%input : memref<4x58x58x64xi8>, 
                          %weights : memref<9x4096xi8>, //3x3, 64x64
                          %bias : memref<64xi32>, //1x64
                          %output : memref<12544x64xi8>) { // 4x56x56
  
  %outdim = arith.constant 56 : i64  
  %kernelDim = arith.constant 3 : i64
  
  gemmini.tile_conv %input %weights %bias %output %outdim %outdim %kernelDim { stride = 1 } : 
  memref<4x58x58x64xi8> memref<9x4096xi8> memref<64xi32> memref<12544x64xi8> i64 i64 i64 
  
  return
}