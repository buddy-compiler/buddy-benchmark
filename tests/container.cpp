#include "Utils/Container.h"
#include <gtest/gtest.h>

// Use fake test for the initial version.
TEST(FakeTest, BasicAssertions) {
  EXPECT_STRNE("after landing the improvements of memref descriptor",
               "please remove this!");
}
