module {
  memref.global "private" constant @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32> = dense<8.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x1x40x40xf32 : memref<1x1x40x40xf32> = dense<4.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_32x128x40xf32 : memref<32x128x40xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_32x40x128xf32 : memref<32x40x128xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x32x40x40xf32 : memref<1x32x40x40xf32> = dense<11.3137083> {alignment = 64 : i64}
  func.func @nofusepartveckenerl(%arg0: tensor<32x40x128xf32>, %arg1: tensor<32x128x40xf32>, %arg2: tensor<1x1x40x40xf32>, %arg3: tensor<1x32x40x128xf32>) {
    %cst = arith.constant 0.0883883461 : f32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant -3.40282347E+38 : f32
    %0 = bufferization.to_memref %arg3 : memref<1x32x40x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg2 : memref<1x1x40x40xf32, strided<[?, ?, ?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg1 : memref<32x128x40xf32, strided<[?, ?, ?], offset: ?>>
    %3 = bufferization.to_memref %arg0 : memref<32x40x128xf32, strided<[?, ?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x40x40xf32>
    affine.for %arg4 = 0 to 32 {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 40 {
          affine.store %cst_0, %alloc[%arg4, %arg5, %arg6] : memref<32x40x40xf32>
        }
      }
    }
    affine.parallel (%arg4) = (0) to (32) {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 40 step 40{
          affine.for %arg7 = 0 to 128 {
            %c1 = arith.constant 0.000000e+00 : f32
            %t1 = vector.transfer_read %3[%arg4, %arg5, %arg7],%c1 {permutation_map = affine_map<(d0,d1,d2)->(0)> } : memref<32x40x128xf32, strided<[?, ?, ?], offset: ?>> , vector<40xf32>
            %c2 = arith.constant 0.000000e+00 : f32
            %t2 = vector.transfer_read %2[%arg4, %arg7, %arg6],%c2 : memref<32x128x40xf32, strided<[?, ?, ?], offset: ?>> , vector<40xf32>
            %c3 = arith.constant 0.000000e+00 : f32
            %t3 = vector.transfer_read %alloc[%arg4, %arg5, %arg6],%c3 : memref<32x40x40xf32> , vector<40xf32>
            %r1 = arith.mulf %t1, %t2 : vector<40xf32>
            %r2 = arith.addf %r1, %t3 : vector<40xf32>
            vector.transfer_write %r2, %alloc[%arg4, %arg5, %arg6] : vector<40xf32>, memref<32x40x40xf32>
          }
        }
      }
    }
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2], [3]] : memref<32x40x40xf32> into memref<1x32x40x40xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            affine.store %cst, %alloc_3[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %expand_shape[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = affine.load %alloc_3[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %7 = arith.mulf %5, %6 : f32
            affine.store %7, %alloc_4[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %alloc_4[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = affine.load %1[%c0, %c0, %arg6, %arg7] : memref<1x1x40x40xf32, strided<[?, ?, ?, ?], offset: ?>>
            %7 = arith.addf %5, %6 : f32
            affine.store %7, %alloc_5[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.store %cst_2, %alloc_6[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
        }
      }
    }
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %alloc_5[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = affine.load %alloc_6[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
            %7 = arith.cmpf ugt, %5, %6 : f32
            %8 = arith.select %7, %5, %6 : f32
            %9 = arith.cmpf uno, %6, %6 : f32
            %10 = arith.select %9, %6, %8 : f32
            affine.store %10, %alloc_6[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
          }
        }
      }
    }
    %expand_shape_7 = memref.expand_shape %alloc_6 [[0], [1], [2, 3]] : memref<1x32x40xf32> into memref<1x32x40x1xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %alloc_5[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = affine.load %expand_shape_7[%c0, %arg5, %arg6, %c0] : memref<1x32x40x1xf32>
            %7 = arith.subf %5, %6 : f32
            affine.store %7, %alloc_8[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %alloc_8[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = math.exp %5 : f32
            affine.store %6, %alloc_9[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.store %cst_0, %alloc_10[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
        }
      }
    }
    affine.for %arg4 = 0 to 1 {
      affine.parallel (%arg5) = (0) to (32) {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 step 40{
            %c1 = arith.constant 0.000000e+00 : f32
            %5 = vector.transfer_read %alloc_9[%arg4, %arg5, %arg6, %arg7],%c1 : memref<1x32x40x40xf32> , vector<40xf32>
            %7 = vector.reduction<add>,%5 : vector<40xf32> into f32
            affine.store %7, %alloc_10[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
            //%5 = affine.load %alloc_9[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            //%6 = affine.load %alloc_10[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
            //%7 = arith.addf %5, %6 : f32
            //affine.store %7, %alloc_10[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
          }
        }
      }
    }
    %expand_shape_11 = memref.expand_shape %alloc_10 [[0], [1], [2, 3]] : memref<1x32x40xf32> into memref<1x32x40x1xf32>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x1xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 1 {
            %5 = affine.load %expand_shape_11[%c0, %arg5, %arg6, %c0] : memref<1x32x40x1xf32>
            %6 = arith.divf %cst_1, %5 : f32
            affine.store %6, %alloc_12[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x1xf32>
          }
        }
      }
    }
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.parallel (%arg5) = (0) to (32) {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 step 40{
            %c1 = arith.constant 0.000000e+00 : f32
            %5 = vector.transfer_read %alloc_9[%c0, %arg5, %arg6, %arg7],%c1 : memref<1x32x40x40xf32> , vector<40xf32>
            %c2 = arith.constant 0.000000e+00 : f32
            %6 = vector.transfer_read %alloc_12[%c0, %arg5, %arg6, %c0],%c2 {permutation_map = affine_map<(d0,d1,d2,d3)->(0)> } : memref<1x32x40x1xf32> , vector<40xf32>
            %7 = arith.mulf %5, %6 : vector<40xf32>
            vector.transfer_write %7,%alloc_13[%arg4, %arg5, %arg6, %arg7] : vector<40xf32>,memref<1x32x40x40xf32>
            //%5 = affine.load %alloc_9[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            //%6 = affine.load %alloc_12[%c0, %arg5, %arg6, %c0] : memref<1x32x40x1xf32>
            //%7 = arith.mulf %5, %6 : f32
            //affine.store %7, %alloc_13[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }
    %collapse_shape = memref.collapse_shape %alloc_13 [[0, 1], [2], [3]] : memref<1x32x40x40xf32> into memref<32x40x40xf32>
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x128xf32>
    memref.copy %0, %alloc_14 : memref<1x32x40x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x32x40x128xf32>
    %collapse_shape_15 = memref.collapse_shape %alloc_14 [[0, 1], [2], [3]] : memref<1x32x40x128xf32> into memref<32x40x128xf32>
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32x40x128xf32>
    affine.for %arg4 = 0 to 32 {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 128 {
          affine.store %cst_0, %alloc_16[%arg4, %arg5, %arg6] : memref<32x40x128xf32>
        }
      }
    }
    // matmul->alloc_16
    affine.parallel (%arg4) = (0) to (32) {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 128 step 128{
          affine.for %arg7 = 0 to 40 {
            %c1 = arith.constant 0.000000e+00 : f32
            %t1 = vector.transfer_read %collapse_shape[%arg4, %arg5, %arg7],%c1 {permutation_map = affine_map<(d0,d1,d2)->(0)> } : memref<32x40x40xf32> , vector<128xf32>
            %c2 = arith.constant 0.000000e+00 : f32
            %t2 = vector.transfer_read %collapse_shape_15[%arg4, %arg7, %arg6],%c2 : memref<32x40x128xf32> , vector<128xf32>
            %c3 = arith.constant 0.000000e+00 : f32
            %t3 = vector.transfer_read %alloc_16[%arg4, %arg5, %arg6],%c3 : memref<32x40x128xf32> , vector<128xf32>
            %r1 = arith.mulf %t1, %t2 : vector<128xf32>
            %r2 = arith.addf %r1, %t3 : vector<128xf32>
            vector.transfer_write %r2, %alloc_16[%arg4, %arg5, %arg6] : vector<128xf32>, memref<32x40x128xf32>
          }
        }
      }
    }
    return
  }
  func.func @main4() {
    %0 = memref.get_global @__constant_32x40x128xf32 : memref<32x40x128xf32>
    %1 = bufferization.to_tensor %0 : memref<32x40x128xf32>
    %2 = memref.get_global @__constant_32x128x40xf32 : memref<32x128x40xf32>
    %3 = bufferization.to_tensor %2 : memref<32x128x40xf32>
    %4 = memref.get_global @__constant_1x1x40x40xf32 : memref<1x1x40x40xf32>
    %5 = bufferization.to_tensor %4 : memref<1x1x40x40xf32>
    %6 = memref.get_global @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32>
    %7 = bufferization.to_tensor %6 : memref<1x32x40x128xf32>
    call @nofusepartveckenerl(%1, %3, %5, %7) : (tensor<32x40x128xf32>, tensor<32x128x40xf32>, tensor<1x1x40x40xf32>, tensor<1x32x40x128xf32>) -> ()
    return
  }
  
}

