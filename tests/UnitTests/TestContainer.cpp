//===- TestContainer.cpp --------------------------------------------------===//
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
// This file implements the tests for the MemRef class.
//
//===----------------------------------------------------------------------===//

#include "Test.h"
#include "Utils/Container.h"

// Fixture for testing the MemRef class.
class MemRefTest : public ::testing::Test {
protected:
  MemRefTest()
      : m1(MemRef<float, 1>({2})), m2(MemRef<float, 2>({2, 3})),
        m3(MemRef<float, 3>({2, 3, 4})), m4(MemRef<float, 4>({2, 3, 4, 5})) {}

  void SetUp() override {
    std::iota(m3.getData(), m3.getData() + 2, 0.);
    std::iota(m2.getData(), m2.getData() + 6, 0.);
    std::iota(m3.getData(), m3.getData() + 24, 0.);
    std::iota(m4.getData(), m4.getData() + 120, 0.);
  }

  MemRef<float, 1> m1;
  MemRef<float, 2> m2;
  MemRef<float, 3> m3;
  MemRef<float, 4> m4;
};

// 1 dimensional memref.
TEST_F(MemRefTest, 1DMemref) {
  intptr_t true_strides[] = {1};
  size_t n_sizes = sizeof(true_strides) / sizeof(intptr_t);
  ASSERT_EQ(m1.getRank(), n_sizes);
  ASSERT_ARRAY_EQ(m1.getStrides(), true_strides, n_sizes);
}

// 2 dimensional memref.
TEST_F(MemRefTest, 2DMemref) {
  intptr_t true_strides[] = {3, 1};
  size_t n_strides = sizeof(true_strides) / sizeof(intptr_t);
  ASSERT_EQ(m2.getRank(), n_strides);
  ASSERT_ARRAY_EQ(m2.getStrides(), true_strides, n_strides);
}

// Transpose a 2d memref.
TEST_F(MemRefTest, Transpose2DMemref) {
  MemRef<float, 2> transposed = m2.transpose();
  // Sizes.
  intptr_t true_sizes[] = {3, 2};
  size_t n_sizes = sizeof(true_sizes) / sizeof(intptr_t);
  ASSERT_EQ(transposed.getRank(), n_sizes);
  ASSERT_ARRAY_EQ(transposed.getSizes(), true_sizes, n_sizes);
  // Strides.
  intptr_t true_strides[] = {2, 1};
  size_t n_strides = sizeof(true_strides) / sizeof(intptr_t);
  ASSERT_EQ(transposed.getRank(), n_strides);
  ASSERT_ARRAY_EQ(transposed.getStrides(), true_strides, n_strides);
  // Data.
  float true_data[] = {0., 3., 1., 4., 2., 5.};
  size_t n_data = sizeof(true_data) / sizeof(float);
  ASSERT_EQ(transposed.getSize(), n_data);
  ASSERT_ARRAY_EQ(transposed.getData(), true_data, n_data);
}

// 3 dimensional memref.
TEST_F(MemRefTest, 3DMemref) {
  intptr_t true_strides[] = {12, 4, 1};
  size_t n_strides = sizeof(true_strides) / sizeof(intptr_t);
  ASSERT_EQ(m3.getRank(), n_strides);
  ASSERT_ARRAY_EQ(m3.getStrides(), true_strides, n_strides);
}

// Transpose a 3d memref.
TEST_F(MemRefTest, Transpose3DMemRef) {
  MemRef<float, 3> transposed = m3.transpose();
  // Sizes.
  intptr_t true_sizes[] = {4, 3, 2};
  size_t n_sizes = sizeof(true_sizes) / sizeof(intptr_t);
  ASSERT_EQ(transposed.getRank(), n_sizes);
  ASSERT_ARRAY_EQ(transposed.getSizes(), true_sizes, n_sizes);
  // Strides.
  intptr_t true_strides[] = {6, 2, 1};
  size_t n_strides = sizeof(true_strides) / sizeof(intptr_t);
  ASSERT_EQ(transposed.getRank(), n_strides);
  ASSERT_ARRAY_EQ(transposed.getStrides(), true_strides, n_strides);
  // Data.
  float true_data[] = {0., 12., 4., 16., 8.,  20., 1., 13., 5., 17., 9.,  21.,
                       2., 14., 6., 18., 10., 22., 3., 15., 7., 19., 11., 23.};
  size_t n_data = sizeof(true_data) / sizeof(float);
  ASSERT_EQ(transposed.getSize(), n_data);
  ASSERT_ARRAY_EQ(transposed.getData(), true_data, n_data);
}

// Convert a 3d memref from CHW to HWC.
TEST_F(MemRefTest, TransposeCHWToHWC) {
  MemRef<float, 3> transposed = m3.transpose({1, 2, 0});
  // Sizes.
  intptr_t true_sizes[] = {3, 4, 2};
  size_t n_sizes = sizeof(true_sizes) / sizeof(intptr_t);
  ASSERT_EQ(transposed.getRank(), n_sizes);
  ASSERT_ARRAY_EQ(transposed.getSizes(), true_sizes, n_sizes);
  // Strides.
  intptr_t true_strides[] = {8, 2, 1};
  size_t n_strides = sizeof(true_strides) / sizeof(intptr_t);
  ASSERT_EQ(transposed.getRank(), n_strides);
  ASSERT_ARRAY_EQ(transposed.getStrides(), true_strides, n_strides);
  // Data.
  float true_data[] = {0., 12., 1., 13., 2., 14., 3., 15., 4.,  16., 5.,  17.,
                       6., 18., 7., 19., 8., 20., 9., 21., 10., 22., 11., 23.};
  size_t n_data = sizeof(true_data) / sizeof(float);
  ASSERT_EQ(transposed.getSize(), n_data);
  ASSERT_ARRAY_EQ(transposed.getData(), true_data, n_data);
}

// 4 dimensional memref.
TEST_F(MemRefTest, 4DMemref) {
  intptr_t true_strides[] = {60, 20, 5, 1};
  size_t n_strides = sizeof(true_strides) / sizeof(intptr_t);
  ASSERT_EQ(m4.getRank(), n_strides);
  ASSERT_ARRAY_EQ(m4.getStrides(), true_strides, n_strides);
}

// Transpose a 4d memref.
TEST_F(MemRefTest, Transpose4DMemRef) {
  MemRef<float, 4> transposed = m4.transpose();
  // Sizes.
  intptr_t true_sizes[] = {5, 4, 3, 2};
  size_t n_sizes = sizeof(true_sizes) / sizeof(intptr_t);
  ASSERT_EQ(transposed.getRank(), n_sizes);
  ASSERT_ARRAY_EQ(transposed.getSizes(), true_sizes, n_sizes);
  // Strides.
  intptr_t true_strides[] = {24, 6, 2, 1};
  size_t n_strides = sizeof(true_strides) / sizeof(intptr_t);
  ASSERT_EQ(transposed.getRank(), n_strides);
  ASSERT_ARRAY_EQ(transposed.getStrides(), true_strides, n_strides);
  // Data.
  float true_data[] = {
      0.,  60., 20., 80., 40., 100., 5.,  65., 25., 85., 45., 105.,
      10., 70., 30., 90., 50., 110., 15., 75., 35., 95., 55., 115.,
      1.,  61., 21., 81., 41., 101., 6.,  66., 26., 86., 46., 106.,
      11., 71., 31., 91., 51., 111., 16., 76., 36., 96., 56., 116.,
      2.,  62., 22., 82., 42., 102., 7.,  67., 27., 87., 47., 107.,
      12., 72., 32., 92., 52., 112., 17., 77., 37., 97., 57., 117.,
      3.,  63., 23., 83., 43., 103., 8.,  68., 28., 88., 48., 108.,
      13., 73., 33., 93., 53., 113., 18., 78., 38., 98., 58., 118.,
      4.,  64., 24., 84., 44., 104., 9.,  69., 29., 89., 49., 109.,
      14., 74., 34., 94., 54., 114., 19., 79., 39., 99., 59., 119.};
  size_t n_data = sizeof(true_data) / sizeof(float);
  ASSERT_EQ(transposed.getSize(), n_data);
  ASSERT_ARRAY_EQ(transposed.getData(), true_data, n_data);
}

// Convert a 4d memref from NCHW to NHWC.
TEST_F(MemRefTest, TransposeNCHWToNHWC) {
  MemRef<float, 4> transposed = m4.transpose({0, 2, 3, 1});
  // Sizes.
  intptr_t true_sizes[] = {2, 4, 5, 3};
  size_t n_sizes = sizeof(true_sizes) / sizeof(intptr_t);
  ASSERT_EQ(transposed.getRank(), n_sizes);
  ASSERT_ARRAY_EQ(transposed.getSizes(), true_sizes, n_sizes);
  // Strides.
  intptr_t true_strides[] = {60, 15, 3, 1};
  size_t n_strides = sizeof(true_strides) / sizeof(intptr_t);
  ASSERT_EQ(transposed.getRank(), n_strides);
  ASSERT_ARRAY_EQ(transposed.getStrides(), true_strides, n_strides);
  // Data.
  float true_data[] = {
      0.,  20., 40.,  1.,  21., 41.,  2.,  22., 42.,  3.,  23., 43.,
      4.,  24., 44.,  5.,  25., 45.,  6.,  26., 46.,  7.,  27., 47.,
      8.,  28., 48.,  9.,  29., 49.,  10., 30., 50.,  11., 31., 51.,
      12., 32., 52.,  13., 33., 53.,  14., 34., 54.,  15., 35., 55.,
      16., 36., 56.,  17., 37., 57.,  18., 38., 58.,  19., 39., 59.,
      60., 80., 100., 61., 81., 101., 62., 82., 102., 63., 83., 103.,
      64., 84., 104., 65., 85., 105., 66., 86., 106., 67., 87., 107.,
      68., 88., 108., 69., 89., 109., 70., 90., 110., 71., 91., 111.,
      72., 92., 112., 73., 93., 113., 74., 94., 114., 75., 95., 115.,
      76., 96., 116., 77., 97., 117., 78., 98., 118., 79., 99., 119.};
  size_t n_data = sizeof(true_data) / sizeof(float);
  ASSERT_EQ(transposed.getSize(), n_data);
  ASSERT_ARRAY_EQ(transposed.getData(), true_data, n_data);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
