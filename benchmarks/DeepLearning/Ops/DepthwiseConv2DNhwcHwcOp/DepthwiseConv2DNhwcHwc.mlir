// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts 

func.func @depthwise_conv_2d_nhwc_hwc(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
  linalg.depthwise_conv_2d_nhwc_hwc 
    {dilations = dense<[1,1]> : tensor<2xi64>, strides = dense<[1,1]> : tensor<2xi64>} 
    ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?xf32>) 
    outs(%arg2 : memref<?x?x?x?xf32>) 
  return
}
