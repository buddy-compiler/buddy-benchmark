module {
  func.func @fpow(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,  // 输入张量的映射
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>   // 输出张量的映射
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg0 : memref<?x?x?x?xf32>) outs(%arg1 : memref<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %fpow = math.powf %in, %in : f32
        linalg.yield %fpow : f32
    }
    return
  }
}
