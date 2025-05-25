//===- MLIRIIRVectorization.mlir ------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file provides the MLIR IIR VECTORIZATION function.
//
//===----------------------------------------------------------------------===//

func.func @iir_vectorization(%in : memref<?xf32>, %filter : memref<?x?xf32>, %out : memref<?xf32>){
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %N = memref.dim %in, %c0 : memref<?xf32>
  %M = memref.dim %filter, %c0: memref<?x?xf32>

  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32
  %vecA1 = vector.splat %f0 : vector<16xf32>
  %vecA2 = vector.splat %f0 : vector<16xf32>
  %vecB0 = vector.splat %f1 : vector<16xf32>
  %vecB1 = vector.splat %f0 : vector<16xf32>
  %vecB2 = vector.splat %f0 : vector<16xf32>

  // biquad params distribution stage
  %B0, %B1, %B2, %A1, %A2 = affine.for %i = 0 to %M iter_args(%vecB0_temp = %vecB0, %vecB1_temp = %vecB1, %vecB2_temp = %vecB2,
                                    %vecA1_temp = %vecA1, %vecA2_temp = %vecA2) -> 
                                    (vector<16xf32>, vector<16xf32>, vector<16xf32>, 
                                    vector<16xf32>, vector<16xf32>){
    %b0 = affine.load %filter[%i, 0] : memref<?x?xf32>
    %b1 = affine.load %filter[%i, 1] : memref<?x?xf32>
    %b2 = affine.load %filter[%i, 2] : memref<?x?xf32>
    %a1 = affine.load %filter[%i, 4] : memref<?x?xf32>
    %a2 = affine.load %filter[%i, 5] : memref<?x?xf32>

    %vecB0_next = vector.insertelement %b0, %vecB0_temp[%i : index]: vector<16xf32>
    %vecB1_next = vector.insertelement %b1, %vecB1_temp[%i : index]: vector<16xf32>
    %vecB2_next = vector.insertelement %b2, %vecB2_temp[%i : index]: vector<16xf32>
    %vecA1_next = vector.insertelement %a1, %vecA1_temp[%i : index]: vector<16xf32>
    %vecA2_next = vector.insertelement %a2, %vecA2_temp[%i : index]: vector<16xf32>
    
    affine.yield %vecB0_next, %vecB1_next, %vecB2_next, %vecA1_next, %vecA2_next : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
  }

  %vecOut = vector.splat %f0 : vector<16xf32>
  %vecS1 = vector.splat %f0 : vector<16xf32>
  %vecS2 = vector.splat %f0 : vector<16xf32>

  // pipeline injection stage
  %Out_stage1, %S1_stage1, %S2_stage1 = affine.for %i = 0 to 15 iter_args(%vecIn_temp = %vecOut, %vecS1_temp = %vecS1, 
                                    %vecS2_temp = %vecS2) -> (vector<16xf32>, vector<16xf32>, 
                                                              vector<16xf32>){
    %input = affine.load %in[%i] : memref<?xf32>
    %vecIn_move_right = vector.shuffle %vecIn_temp, %vecIn_temp[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] : vector<16xf32>, vector<16xf32>
    %vecIn_next = vector.insertelement %input, %vecIn_move_right[%c0 : index]: vector<16xf32>
    %vecOut_next = vector.fma %B0, %vecIn_next, %vecS1_temp : vector<16xf32>

    %vecS1_part1 = vector.fma %B1, %vecIn_next, %vecS2_temp : vector<16xf32>
    %vecS1_part2 = arith.mulf %A1, %vecOut_next : vector<16xf32>
    %vecS1_next = arith.subf %vecS1_part1, %vecS1_part2 : vector<16xf32>

    %vecS2_part1 = arith.mulf %B2, %vecIn_next : vector<16xf32>
    %vecS2_part2 = arith.mulf %A2, %vecOut_next : vector<16xf32>
    %vecS2_next = arith.subf %vecS2_part1, %vecS2_part2 : vector<16xf32>

    affine.yield %vecOut_next, %vecS1_next, %vecS2_next : vector<16xf32>, vector<16xf32>, vector<16xf32>
  }

  %i15 = arith.constant 15 : index
  %upperbound = arith.subi %N, %i15 : index

  // pipeline process stage
  %Out_stage2, %S1_stage2, %S2_stage2 = affine.for %i = 0 to %upperbound iter_args(%vecIn_temp = %Out_stage1, %vecS1_temp = %S1_stage1, 
                                             %vecS2_temp = %S2_stage1) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>){
    %input = affine.load %in[%i + 15] : memref<?xf32>
    %vecIn_move_right = vector.shuffle %vecIn_temp, %vecIn_temp[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] : vector<16xf32>, vector<16xf32>
    %vecIn_next = vector.insertelement %input, %vecIn_move_right[%c0 : index]: vector<16xf32>
    %vecOut_next = vector.fma %B0, %vecIn_next, %vecS1_temp : vector<16xf32>
    %output = vector.extractelement %vecOut_next[%i15 : index] : vector<16xf32>
    affine.store %output, %out[%i] : memref<?xf32>

    %vecS1_part1 = vector.fma %B1, %vecIn_next, %vecS2_temp : vector<16xf32>
    %vecS1_part2 = arith.mulf %A1, %vecOut_next : vector<16xf32>
    %vecS1_next = arith.subf %vecS1_part1, %vecS1_part2 : vector<16xf32>

    %vecS2_part1 = arith.mulf %B2, %vecIn_next : vector<16xf32>
    %vecS2_part2 = arith.mulf %A2, %vecOut_next : vector<16xf32>
    %vecS2_next = arith.subf %vecS2_part1, %vecS2_part2 : vector<16xf32>

    affine.yield %vecOut_next, %vecS1_next, %vecS2_next : vector<16xf32>, vector<16xf32>, vector<16xf32>
  }

  // pipeline tail ending stage
  affine.for %i = %upperbound to %N iter_args(%vecIn_temp = %Out_stage2, %vecS1_temp = %S1_stage2, 
                                              %vecS2_temp = %S2_stage2) -> (vector<16xf32>, vector<16xf32>, 
                                                                                     vector<16xf32>){
    %vecIn_move_right = vector.shuffle %vecIn_temp, %vecIn_temp[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] : vector<16xf32>, vector<16xf32>
    %vecIn_next = vector.insertelement %f0, %vecIn_move_right[%c0 : index]: vector<16xf32>
    %vecOut_next = vector.fma %B0, %vecIn_next, %vecS1_temp : vector<16xf32>
    %output = vector.extractelement %vecOut_next[%i15 : index] : vector<16xf32>
    affine.store %output, %out[%i] : memref<?xf32>

    %vecS1_part1 = vector.fma %B1, %vecIn_next, %vecS2_temp : vector<16xf32>
    %vecS1_part2 = arith.mulf %A1, %vecOut_next : vector<16xf32>
    %vecS1_next = arith.subf %vecS1_part1, %vecS1_part2 : vector<16xf32>

    %vecS2_part1 = arith.mulf %B2, %vecIn_next : vector<16xf32>
    %vecS2_part2 = arith.mulf %A2, %vecOut_next : vector<16xf32>
    %vecS2_next = arith.subf %vecS2_part1, %vecS2_part2 : vector<16xf32>
    
    affine.yield %vecOut_next, %vecS1_next, %vecS2_next : vector<16xf32>, vector<16xf32>, vector<16xf32>
  }

  return
}
