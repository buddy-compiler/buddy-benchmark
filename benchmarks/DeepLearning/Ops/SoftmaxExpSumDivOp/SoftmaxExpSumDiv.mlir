module {
  func.func @softmaxexpsumdiv(%input: memref<1x?x?xf32>, %output: memref<1x?x?xf32>) {
    linalg.generic
      {indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input: memref<1x?x?xf32>)
      outs(%output: memref<1x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %cst_2711 = arith.constant 1.000000e+00 : f32
        %4105 = arith.negf %in : f32
        %4106 = math.exp %4105 : f32
        %4107 = arith.addf %4106, %cst_2711 : f32
        %4108 = arith.divf %cst_2711, %4107 : f32
        linalg.yield %4108 : f32
    }

    return
  }
}
