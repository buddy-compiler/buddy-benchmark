func.func @conv_2d(%arg0: memref<10x10xf32>, %arg1: memref<3x3xf32>, %arg2: memref<8x8xf32>) {
    linalg.conv_2d ins (%arg0, %arg1: memref<10x10xf32>, memref<3x3xf32>)
                  outs (%arg2: memref<8x8xf32>)
    return
  }