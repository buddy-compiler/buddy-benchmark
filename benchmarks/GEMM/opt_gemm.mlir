#map0 = affine_map<()[s0] -> (s0 ceildiv 120)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 110)>
#map2 = affine_map<()[s0] -> (s0 ceildiv 64)>
#map3 = affine_map<(d0) -> (d0 * 16)>
#map4 = affine_map<(d0)[s0] -> (d0 * 16 + 16, s0 floordiv 4)>
#map5 = affine_map<(d0)[s0] -> (120, d0 * -120 + s0)>
#map6 = affine_map<(d0)[s0] -> (110, d0 * -110 + s0)>
#map7 = affine_map<(d0, d1) -> (d0 * 110 + d1)>
#map8 = affine_map<(d0, d1) -> (d0 * 120 + d1)>
#map9 = affine_map<(d0, d1) -> (d0 * 16 + d1 * 4)>
#map10 = affine_map<(d0, d1) -> (0)>
module {
  func.func @gemm(%arg0: memref<?x?xf64>, %arg1: memref<?x?xf64>, %arg2: memref<?x?xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.dim %arg0, %c0 : memref<?x?xf64>
    %1 = memref.dim %arg0, %c1 : memref<?x?xf64>
    %2 = memref.dim %arg1, %c1 : memref<?x?xf64>
    affine.for %arg3 = 0 to #map0()[%1] {
      affine.for %arg4 = 0 to #map1()[%0] {
// 这里我们应该对arg0进行局部操作
        %local_a = memref.alloc() : memref<120x110xf64>
        affine.for %arg_local_7 = 0 to min #map5(%arg3)[%1] {
          affine.for %arg_local_8 = 0 to min #map6(%arg4)[%0] {
            %local_3 = affine.apply #map7(%arg4, %arg_local_8) // 范围？~110
            %local_4 = affine.apply #map8(%arg3, %arg_local_7) // 范围？~120
            %t_a = memref.load %arg0[%local_3, %local_4] : memref<?x?xf64>
            memref.store %t_a, %local_a[%arg_local_7, %arg_local_8] : memref<120x110xf64>
          }
        }
        affine.for %arg5 = 0 to #map2()[%2] {
          affine.for %arg6 = #map3(%arg5) to min #map4(%arg5)[%2] {
// 这里我们应该对arg1进行局部操作
            %local_b = memref.alloc() : memref<4x120xvector<4xf64>>
            affine.for %arg_local_7 = 0 to min #map5(%arg3)[%1] {
                affine.for %arg_local_9 = 0 to 4 {
                  %local_4 = affine.apply #map8(%arg3, %arg_local_7) // 范围？~120
                  %local_5 = affine.apply #map9(%arg6, %arg_local_9)
                  %local_7 = vector.load %arg1[%local_4, %local_5] : memref<?x?xf64>, vector<4xf64>
                  vector.store %local_7, %local_b[%arg_local_9, %arg_local_7] : memref<4x120xvector<4xf64>>, vector<4xf64>
 		}
	    }
            affine.for %arg8 = 0 to min #map6(%arg4)[%0] {
              %local_c = memref.alloc() : memref<4xvector<4xf64>>
	      affine.for %arg_local_9 = 0 to 4 {
                  %local_3 = affine.apply #map7(%arg4, %arg8) // 范围？~110
                  %local_5 = affine.apply #map9(%arg6, %arg_local_9)
                  %local_8 = vector.load %arg2[%local_3, %local_5] : memref<?x?xf64>, vector<4xf64> // 现在我们要搞他
                  vector.store %local_8, %local_c[%arg_local_9] : memref<4xvector<4xf64>>, vector<4xf64>
	      }
              affine.for %arg7 = 0 to min #map5(%arg3)[%1] {
                affine.for %arg9 = 0 to 4 {
                  %6 = vector.transfer_read %local_a[%arg7, %arg8], %cst {permutation_map = #map10} : memref<120x110xf64>, vector<4xf64>
		  %7 = vector.load %local_b[%arg9, %arg7] : memref<4x120xvector<4xf64>>, vector<4xf64>
		  %8 = vector.load %local_c[%arg9] : memref<4xvector<4xf64>>, vector<4xf64>
                  %9 = arith.mulf %6, %7 : vector<4xf64>
                  %10 = arith.addf %8, %9 : vector<4xf64>
                  vector.store %10, %local_c[%arg9] : memref<4xvector<4xf64>>, vector<4xf64>
                }
              }
	      affine.for %arg_local_9 = 0 to 4 {
                  %local_3 = affine.apply #map7(%arg4, %arg8) // 范围？~110
                  %local_5 = affine.apply #map9(%arg6, %arg_local_9)
                  %local_8 = vector.load %arg2[%local_3, %local_5] : memref<?x?xf64>, vector<4xf64> // 现在我们要搞他
                  vector.transfer_write %local_8, %arg2[%local_3, %local_5] : vector<4xf64>, memref<?x?xf64>
	      }
	      memref.dealloc %local_c : memref<4xvector<4xf64>>
            }
	    memref.dealloc %local_b : memref<4x120xvector<4xf64>>
          }
        }
	memref.dealloc %local_a : memref<120x110xf64>
      }
    }
    return
  }
}

