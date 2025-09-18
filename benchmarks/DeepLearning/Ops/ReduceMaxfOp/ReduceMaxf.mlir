module {
  func.func @maxf(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
    %init = arith.constant -3.4028235e+38 : f32  // Representing negative infinity for float
    linalg.reduce  ins(%arg0 : memref<?x?x?x?xf32>) outs(%arg1 : memref<?x?x?xf32>) dimensions = [3] 
      (%in: f32, %out:f32) {
        %maxf = arith.maximumf %in, %init: f32
        linalg.yield %maxf : f32
    } 
    return
  }
}
