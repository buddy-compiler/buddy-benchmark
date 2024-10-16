func.func @depthwise_conv_2d_nhwc_hwc(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
  linalg.depthwise_conv_2d_nhwc_hwc 
    ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?xf32>) 
    outs(%arg2 : memref<?x?x?x?xf32>) 
  return
}
