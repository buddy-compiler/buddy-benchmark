#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
module {
  func.func @base(%arg0: memref<1x512x4096xf32>, %arg1: memref<1x512x1024xf32>) {
    linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%arg0 : memref<1x512x4096xf32>) outs(%arg1 : memref<1x512x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_2372 = arith.constant 4.096000e+03 : f32
      %4230 = arith.divf %in, %cst_2372 : f32
      %4231 = arith.addf %4230, %out : f32
      linalg.yield %4231 : f32
    }
    return
  }
}