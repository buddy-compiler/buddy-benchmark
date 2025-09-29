#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <cmath>

typedef short DCTELEM;

extern void fdct_ifast(DCTELEM *data);

class JfdctfstTest : public ::testing::Test {
protected:
    void SetUp() override {

        for (int i = 0; i < 64; i++) {
            test_block[i] = 0;
        }

        for (int i = 0; i < 64; i++) {
            uniform_block[i] = 128;
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                gradient_block[i * 8 + j] = i * 16 + j * 2;
            }
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                checkerboard_block[i * 8 + j] = ((i + j) % 2) ? 255 : 0;
            }
        }

        for (int i = 0; i < 64; i++) {
            impulse_block[i] = 0;
        }
        impulse_block[0] = 1024;
    }

    short test_block[64];
    short uniform_block[64];
    short gradient_block[64];
    short checkerboard_block[64];
    short impulse_block[64];
};

TEST_F(JfdctfstTest, HandlesZeroBlock) {

    fdct_ifast(test_block);

    for (int i = 0; i < 64; i++) {
        EXPECT_EQ(test_block[i], 0);
    }
}

TEST_F(JfdctfstTest, HandlesUniformBlock) {

    fdct_ifast(uniform_block);

    EXPECT_NE(uniform_block[0], 0);
    EXPECT_GT(abs(uniform_block[0]), 500);

    for (int i = 1; i < 64; i++) {
        EXPECT_NEAR(uniform_block[i], 0, 10);
    }
}

TEST_F(JfdctfstTest, HandlesGradientBlock) {
    fdct_ifast(gradient_block);

    EXPECT_NE(gradient_block[0], 0);
    EXPECT_NE(gradient_block[1], 0);
    EXPECT_NE(gradient_block[8], 0);

    EXPECT_LT(abs(gradient_block[63]), abs(gradient_block[0]));
    EXPECT_LT(abs(gradient_block[56]), abs(gradient_block[1]));
}

TEST_F(JfdctfstTest, HandlesCheckerboardPattern) {

    fdct_ifast(checkerboard_block);

    EXPECT_GT(abs(checkerboard_block[63]), 400);

    EXPECT_EQ(checkerboard_block[7], 0);
    EXPECT_EQ(checkerboard_block[56], 0);

    bool has_significant_coeffs = false;
    for (int i = 0; i < 64; i++) {
        if (abs(checkerboard_block[i]) > 400) {
            has_significant_coeffs = true;
            break;
        }
    }
    EXPECT_TRUE(has_significant_coeffs);
}

TEST_F(JfdctfstTest, OutputRangeTest) {

    for (int i = 0; i < 64; i++) {
        test_block[i] = (i % 2) ? 255 : 0;
    }

    fdct_ifast(test_block);

    for (int i = 0; i < 64; i++) {
        EXPECT_GE(test_block[i], -16384);
        EXPECT_LE(test_block[i], 16384);
    }
}

TEST_F(JfdctfstTest, DCCoefficientTest) {

    short dc_test[64];
    int sum = 0;

    for (int i = 0; i < 64; i++) {
        dc_test[i] = i + 1;
        sum += dc_test[i];
    }

    fdct_ifast(dc_test);

    EXPECT_GT(abs(dc_test[0]), abs(sum) / 16);
}

TEST_F(JfdctfstTest, LinearityTest) {

    short block1[64], block2[64];

    for (int i = 0; i < 64; i++) {
        block1[i] = i * 4;
        block2[i] = i * 8;
    }

    fdct_ifast(block1);
    fdct_ifast(block2);

    for (int i = 1; i < 64; i++) {
        if (abs(block1[i]) > 10) {
            double ratio = (double)block2[i] / block1[i];
            EXPECT_NEAR(ratio, 2.0, 0.3);
        }
    }
}

TEST_F(JfdctfstTest, ImpulseResponseTest) {

    fdct_ifast(impulse_block);

    EXPECT_GT(abs(impulse_block[0]), 50);

    int non_zero_count = 0;
    for (int i = 1; i < 64; i++) {
        if (abs(impulse_block[i]) > 5) {
            non_zero_count++;
        }
    }
    EXPECT_GT(non_zero_count, 8);
}

TEST_F(JfdctfstTest, SymmetryTest) {

    short symmetric_block[64];

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int val = (j < 4) ? j * 32 : (7 - j) * 32;
            symmetric_block[i * 8 + j] = val;
        }
    }

    fdct_ifast(symmetric_block);

    EXPECT_NEAR(symmetric_block[1], 0, 20);
}

TEST_F(JfdctfstTest, EnergyConservationTest) {

    short energy_block[64];
    long long input_energy = 0, output_energy = 0;

    for (int i = 0; i < 64; i++) {
        energy_block[i] = (i % 8) * 32;
        input_energy += (long long)energy_block[i] * energy_block[i];
    }

    fdct_ifast(energy_block);

    for (int i = 0; i < 64; i++) {
        output_energy += (long long)energy_block[i] * energy_block[i];
    }

    EXPECT_GT(output_energy, input_energy / 100);
    EXPECT_LT(output_energy, input_energy * 100);
}

TEST_F(JfdctfstTest, ReproducibilityTest) {

    short block1[64], block2[64];

    for (int i = 0; i < 64; i++) {
        block1[i] = block2[i] = (i * 7) % 256;
    }

    fdct_ifast(block1);
    fdct_ifast(block2);

    for (int i = 0; i < 64; i++) {
        EXPECT_EQ(block1[i], block2[i]);
    }
}
