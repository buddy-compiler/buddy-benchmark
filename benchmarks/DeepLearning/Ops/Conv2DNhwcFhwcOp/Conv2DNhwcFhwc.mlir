// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts 

func.func @conv_2d_nhwc_fhwc(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
    linalg.conv_2d_nhwc_fhwc 
    ins(%input, %filter : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) 
    outs(%output : memref<?x?x?x?xf32>) 
    return
}
