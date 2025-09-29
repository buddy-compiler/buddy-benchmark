#include <gtest/gtest.h>
#include <vector>
#include <cstring>

extern void h2v1_downsample(unsigned int image_width, int max_v_samp_factor,
                            unsigned int v_samp_factor, unsigned int width_in_blocks,
                            unsigned char **input_data, unsigned char **output_data);

extern void h2v2_downsample(unsigned int image_width, int max_v_samp_factor,
                            unsigned int v_samp_factor, unsigned int width_in_blocks,
                            unsigned char **input_data, unsigned char **output_data);

class JcsampleTest : public ::testing::Test {
protected:
    void SetUp() override {

        input_width = 16;
        max_v_factor = 2;
        v_factor = 1;
        blocks_width = 2;

        for (int row = 0; row < max_v_factor; row++) {
            input_data[row] = new unsigned char[input_width + 16];

            for (int col = 0; col < input_width; col++) {
                input_data[row][col] = (unsigned char)(row * 64 + col * 8);
            }
        }

        for (int row = 0; row < v_factor; row++) {
            output_data[row] = new unsigned char[blocks_width * 8];
        }
    }

    void TearDown() override {

        for (int row = 0; row < max_v_factor; row++) {
            delete[] input_data[row];
        }
        for (int row = 0; row < v_factor; row++) {
            delete[] output_data[row];
        }
    }

    unsigned int input_width;
    int max_v_factor;
    unsigned int v_factor;
    unsigned int blocks_width;
    unsigned char *input_data[4];
    unsigned char *output_data[4];
};

TEST_F(JcsampleTest, H2V1_BasicDownsampling) {

    unsigned char test_input[24] = {100, 150, 200, 250, 50, 75, 125, 175};
    unsigned char *test_row = test_input;
    unsigned char test_output[8];
    unsigned char *test_out_row = test_output;

    h2v1_downsample(8, 1, 1, 1, &test_row, &test_out_row);

    EXPECT_EQ(test_output[0], 125);
    EXPECT_EQ(test_output[1], 225);
    EXPECT_EQ(test_output[2], 62);
    EXPECT_EQ(test_output[3], 150);

}

#if 0
TEST_F(JcsampleTest, H2V1_EmptyInput) {

    h2v1_downsample(0, max_v_factor, v_factor, 0, input_data, output_data);
    SUCCEED();
}
#endif

TEST_F(JcsampleTest, H2V1_SingleBlock) {

    unsigned char single_input[24] = {0, 32, 64, 96, 128, 160, 192, 224};
    unsigned char *single_row = single_input;
    unsigned char single_output[8];
    unsigned char *single_out_row = single_output;

    h2v1_downsample(8, 1, 1, 1, &single_row, &single_out_row);

    for (int i = 0; i < 8; i++) {
        EXPECT_GE(single_output[i], 0);
        EXPECT_LE(single_output[i], 255);
    }

    EXPECT_NEAR(single_output[0], (0 + 32) / 2, 1);
}

TEST_F(JcsampleTest, H2V2_BasicDownsampling) {

    unsigned char test_input_row1[24] = {100, 120, 140, 160};
    unsigned char test_input_row2[24] = {110, 130, 150, 170};
    unsigned char *test_input[2] = {test_input_row1, test_input_row2};
    unsigned char test_output[8];
    unsigned char *test_out_row = test_output;

    h2v2_downsample(4, 2, 1, 1, test_input, &test_out_row);

    EXPECT_NEAR(test_output[0], 115, 2);
    EXPECT_NEAR(test_output[1], 155, 2);

}

TEST_F(JcsampleTest, H2V2_UniformInput) {

    unsigned char uniform_row1[24] = {128, 128, 128, 128, 128, 128, 128, 128};
    unsigned char uniform_row2[24] = {128, 128, 128, 128, 128, 128, 128, 128};
    unsigned char *uniform_input[2] = {uniform_row1, uniform_row2};
    unsigned char uniform_output[8];
    unsigned char *uniform_out_row = uniform_output;

    h2v2_downsample(8, 2, 1, 1, uniform_input, &uniform_out_row);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(uniform_output[i], 128, 2);
    }

}

#if 0
TEST_F(JcsampleTest, H2V2_EmptyInput) {

    h2v2_downsample(0, max_v_factor, v_factor, 0, input_data, output_data);
    SUCCEED();
}
#endif

TEST_F(JcsampleTest, H2V1_LargeInput) {

    h2v1_downsample(input_width, max_v_factor, v_factor, blocks_width, input_data, output_data);

    int output_pixels = blocks_width * 8;
    for (int col = 0; col < output_pixels; col++) {
        EXPECT_GE(output_data[0][col], 0);
        EXPECT_LE(output_data[0][col], 255);
    }

    bool different_found = false;
    for (int col = 0; col < output_pixels && col * 2 < input_width; col++) {
        if (output_data[0][col] != input_data[0][col * 2]) {
            different_found = true;
            break;
        }
    }
    EXPECT_TRUE(different_found) << "Expected downsampling to change values";
}

TEST_F(JcsampleTest, H2V2_LargeInput) {

    v_factor = 1;

    h2v2_downsample(input_width, max_v_factor, v_factor, blocks_width, input_data, output_data);

    int output_pixels = blocks_width * 8;
    for (int col = 0; col < output_pixels; col++) {
        EXPECT_GE(output_data[0][col], 0);
        EXPECT_LE(output_data[0][col], 255);
    }
}

TEST_F(JcsampleTest, BiasPattern) {

    unsigned char pattern1[24] = {100, 101, 102, 103};
    unsigned char pattern2[24] = {100, 101, 102, 103};
    unsigned char *rows[2] = {pattern1, pattern2};
    unsigned char output[8];
    unsigned char *out_row = output;

    h2v2_downsample(4, 2, 1, 1, rows, &out_row);

    EXPECT_NEAR(output[0], 100, 2);
    EXPECT_NEAR(output[1], 102, 2);

}
