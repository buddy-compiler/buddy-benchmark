//===- BuddyCorr2D.mlir ---------------------------------------------------===//
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
// This file provides the Buddy Corr2D function.
//
//===----------------------------------------------------------------------===//

func @corr_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, 
                               %outputImage : memref<?x?xf32>, %centerX : index, 
                               %centerY : index, %constantValue : f32) {
  dip.corr_2d CONSTANT_PADDING %inputImage, %kernel, %outputImage, %centerX, 
              %centerY, %constantValue : 
              memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, 
              index, f32
  return
}
