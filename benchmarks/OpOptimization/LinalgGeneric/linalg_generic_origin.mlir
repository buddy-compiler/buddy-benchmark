#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0 + 256, 4096)>
func.func @origin(%arg0: memref<4096x4096xf32>, %arg1: memref<4096xf32>) {
  linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins (%arg0: memref<4096x4096xf32>) outs(%arg1: memref<4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %out : f32
    linalg.yield %2 : f32 
  }
  return
}