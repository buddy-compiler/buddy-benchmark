module {
  func.func @subf(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,  
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, 
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>  
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) outs(%arg2 : memref<?x?x?x?xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out: f32):
        %sub = arith.subf %in0, %in1 : f32
        linalg.yield %sub : f32
    }
    return
  }
}
