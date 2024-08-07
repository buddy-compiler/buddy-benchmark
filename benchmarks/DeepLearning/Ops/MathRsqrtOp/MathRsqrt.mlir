module {
  func.func @rsqrt(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,  // 输入张量的映射
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>   // 输出张量的映射
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg0 : memref<?x?x?x?xf32>) outs(%arg1 : memref<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %rsqrt = math.rsqrt %in : f32
        linalg.yield %rsqrt : f32
    }
    return
  }
}
