#include "Test.h"
#include "Utils/Container.h"

// Fixture for testsing the MemRef class.
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

// 2 dimensional memref.
TEST_F(MemRefTest, 2DMemref) {
  intptr_t true_strides[] = {3, 1};
  ASSERT_ARRAY_EQ(m2.getStrides(), true_strides);
}

// Transpose a 2d memref.
TEST_F(MemRefTest, Transpose2DMemref) {
  MemRef<float, 2> transposed = m2.transpose();
  // Sizes.
  intptr_t true_sizes[] = {3, 2};
  ASSERT_ARRAY_EQ(transposed.getSizes(), true_sizes);
  // Strides.
  intptr_t true_strides[] = {2, 1};
  ASSERT_ARRAY_EQ(transposed.getStrides(), true_strides);
  // Data.
  float true_data[] = {0., 3., 1., 4., 2., 5.};
  ASSERT_ARRAY_EQ(transposed.getData(), true_data);
}

// 3 dimensional memref.
TEST_F(MemRefTest, 3DMemref) {
  intptr_t true_strides[] = {12, 4, 1};
  ASSERT_ARRAY_EQ(m3.getStrides(), true_strides);
}

// Transpose a 3d memref.
TEST_F(MemRefTest, Transpose3DMemRef) {
  MemRef<float, 3> transposed = m3.transpose();
  // Sizes.
  intptr_t true_sizes[] = {4, 3, 2};
  ASSERT_ARRAY_EQ(transposed.getSizes(), true_sizes);
  // Strides.
  intptr_t true_strides[] = {6, 2, 1};
  ASSERT_ARRAY_EQ(transposed.getStrides(), true_strides);
  // Data.
  float true_data[] = {0., 12., 4., 16., 8.,  20., 1., 13., 5., 17., 9.,  21.,
                       2., 14., 6., 18., 10., 22., 3., 15., 7., 19., 11., 23.};
  ASSERT_ARRAY_EQ(transposed.getData(), true_data);
}

// CHW to HWC.
TEST_F(MemRefTest, TransposeCHWToHWC) {
  MemRef<float, 3> transposed = m3.transpose({1, 2, 0});
  // Sizes.
  intptr_t true_sizes[] = {3, 4, 2};
  ASSERT_ARRAY_EQ(transposed.getSizes(), true_sizes);
  // Strides.
  intptr_t true_strides[] = {8, 2, 1};
  ASSERT_ARRAY_EQ(transposed.getStrides(), true_strides);
  // Data.
  float true_data[] = {0., 12., 1., 13., 2., 14., 3., 15., 4.,  16., 5.,  17.,
                       6., 18., 7., 19., 8., 20., 9., 21., 10., 22., 11., 23.};
  ASSERT_ARRAY_EQ(transposed.getData(), true_data);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
