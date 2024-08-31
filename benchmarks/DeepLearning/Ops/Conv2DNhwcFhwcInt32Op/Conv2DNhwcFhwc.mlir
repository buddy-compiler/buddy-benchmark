// generate from iree processed mobilenet mlir file
func.func @conv_2d_nhwc_fhwc(%input: memref<?x?x?x?xi32>, %filter: memref<?x?x?x?xi32>, %output: memref<?x?x?x?xi32>) {
    linalg.conv_2d_nhwc_fhwc 
    ins(%input, %filter : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) 
    outs(%output : memref<?x?x?x?xi32>) 
    return
}
