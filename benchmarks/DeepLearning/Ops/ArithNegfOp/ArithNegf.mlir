module {
  func.func @negf(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,  
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>   
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg0 : memref<?x?x?x?xf32>) outs(%arg1 : memref<?x?x?x?xf32>) {
      ^bb0(%in0: f32, %out: f32):
        %neg = arith.negf %in0 : f32
        linalg.yield %neg : f32
    }
    return
  }
}
