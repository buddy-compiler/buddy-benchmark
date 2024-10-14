module {
  func.func @addf(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
    %init = arith.constant 0.0 : f32
    linalg.reduce  ins(%arg0 : memref<?x?x?x?xf32>) outs(%arg1 : memref<?x?x?xf32>) dimensions = [3] 
      (%in: f32, %out:f32) {
        %addf = arith.addf %in, %init: f32
        linalg.yield %addf : f32
    } 
    return
  }
}
