func.func @conv_2d_nhwc_fhwc(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
    linalg.conv_2d_nhwc_fhwc 
    ins(%input, %filter : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) 
    outs(%output : memref<?x?x?x?xf32>) 
    return
}
