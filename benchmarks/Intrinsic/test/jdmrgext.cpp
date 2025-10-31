#include <gtest/gtest.h>
#include <vector>
#include <cstring>

typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;
typedef unsigned int JDIMENSION;

extern void h2v1_merged_upsample(JDIMENSION output_width, JSAMPIMAGE input_buf,
                                 JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf);

extern void h2v2_merged_upsample(JDIMENSION output_width, JSAMPIMAGE input_buf,
                                 JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf);

class JdmrgextTest : public ::testing::Test {
protected:
    void SetUp() override {

        input_width = 8;
        output_width = 16;
        height = 4;

        for (int row = 0; row < height; row++) {
            ycc_planes[0][row] = new unsigned char[output_width];
        }

        for (int plane = 1; plane < 3; plane++) {
            for (int row = 0; row < height; row++) {
                ycc_planes[plane][row] = new unsigned char[output_width / 2];
            }
        }

        for (int row = 0; row < height; row++) {
            rgb_data[row] = new unsigned char[output_width * 3];
        }

        for (int row = 0; row < height; row++) {

            for (int col = 0; col < output_width; col++) {
                ycc_planes[0][row][col] = (unsigned char)(row * 20 + col * 8);
            }

            for (int col = 0; col < output_width / 2; col++) {
                ycc_planes[1][row][col] = 128;
                ycc_planes[2][row][col] = 128;
            }
        }
    }

    void TearDown() override {

        for (int plane = 0; plane < 3; plane++) {
            for (int row = 0; row < height; row++) {
                delete[] ycc_planes[plane][row];
            }
        }
        for (int row = 0; row < height; row++) {
            delete[] rgb_data[row];
        }
    }

    int input_width, output_width, height;
    unsigned char *ycc_planes[3][4];
    unsigned char **ycc_input[3] = {ycc_planes[0], ycc_planes[1], ycc_planes[2]};
    unsigned char *rgb_data[4];
};

TEST_F(JdmrgextTest, H2V1_MergedUpsampleBasic) {

    unsigned char y_in[8] = {50, 100, 150, 200, 50, 100, 150, 200};

    unsigned char cb_in[4] = {128, 128, 128, 128};
    unsigned char cr_in[4] = {128, 128, 128, 128};
    unsigned char *y_ptr = y_in;
    unsigned char *cb_ptr = cb_in;
    unsigned char *cr_ptr = cr_in;
    unsigned char **test_input[3] = {&y_ptr, &cb_ptr, &cr_ptr};
    unsigned char rgb_out[24];
    unsigned char *test_output = rgb_out;

    h2v1_merged_upsample(8, test_input, 0, &test_output);

    EXPECT_NEAR(rgb_out[0], 50, 10);
    EXPECT_NEAR(rgb_out[1], 50, 10);
    EXPECT_NEAR(rgb_out[2], 50, 10);

    EXPECT_NEAR(rgb_out[3], 100, 10);
    EXPECT_NEAR(rgb_out[4], 100, 10);
    EXPECT_NEAR(rgb_out[5], 100, 10);

    for (int i = 0; i < 24; i++) {
        EXPECT_GE(rgb_out[i], 0);
        EXPECT_LE(rgb_out[i], 255);
    }
}

TEST_F(JdmrgextTest, H2V1_EmptyInput) {

    h2v1_merged_upsample(0, ycc_input, 0, rgb_data);
    SUCCEED();
}

TEST_F(JdmrgextTest, H2V1_ColorConversion) {

    unsigned char y_val[4] = {128, 128, 128, 128};

    unsigned char cb_high[2] = {200, 200};
    unsigned char cr_low[2] = {128, 128};
    unsigned char *y_ptr = y_val;
    unsigned char *cb_ptr = cb_high;
    unsigned char *cr_ptr = cr_low;
    unsigned char **color_input[3] = {&y_ptr, &cb_ptr, &cr_ptr};
    unsigned char rgb_out[12];
    unsigned char *color_output = rgb_out;

    h2v1_merged_upsample(4, color_input, 0, &color_output);

    for (int pixel = 0; pixel < 4; pixel++) {
        int base = pixel * 3;
        EXPECT_GT(rgb_out[base + 2], rgb_out[base + 0]);
        EXPECT_GT(rgb_out[base + 2], rgb_out[base + 1]);
    }
}

TEST_F(JdmrgextTest, H2V2_MergedUpsampleBasic) {

    unsigned char y_row1[4] = {100, 100, 200, 200};
    unsigned char y_row2[4] = {100, 100, 200, 200};
    unsigned char *y_rows[2] = {y_row1, y_row2};
    unsigned char cb_in[2] = {128, 128};
    unsigned char cr_in[2] = {128, 128};
    unsigned char *cb_ptr = cb_in;
    unsigned char *cr_ptr = cr_in;
    unsigned char **test_input[3] = {y_rows, &cb_ptr, &cr_ptr};
    unsigned char rgb_out1[12];
    unsigned char rgb_out2[12];
    unsigned char *test_output[2] = {rgb_out1, rgb_out2};

    h2v2_merged_upsample(4, test_input, 0, test_output);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(rgb_out1[i], rgb_out1[i + 3], 5);
        EXPECT_NEAR(rgb_out1[i], rgb_out2[i], 5);
        EXPECT_NEAR(rgb_out1[i], rgb_out2[i + 3], 5);
    }

    for (int i = 6; i < 9; i++) {
        EXPECT_GT(rgb_out1[i], rgb_out1[i - 6]);
        EXPECT_NEAR(rgb_out1[i], rgb_out1[i + 3], 5);
        EXPECT_NEAR(rgb_out1[i], rgb_out2[i], 5);
        EXPECT_NEAR(rgb_out1[i], rgb_out2[i + 3], 5);
    }
}

TEST_F(JdmrgextTest, H2V2_EmptyInput) {

    h2v2_merged_upsample(0, ycc_input, 0, rgb_data);
    SUCCEED();
}

TEST_F(JdmrgextTest, H2V1_LargeInput) {

    for (int row = 0; row < height; row++) {
        h2v1_merged_upsample(output_width, ycc_input, row, &rgb_data[row]);
    }

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < output_width * 3; col++) {
            EXPECT_GE(rgb_data[row][col], 0);
            EXPECT_LE(rgb_data[row][col], 255);
        }
    }

    for (int row = 0; row < height; row++) {
        for (int pixel = 0; pixel < output_width; pixel++) {
            int base = pixel * 3;
            int expected_y = row * 20 + pixel * 8;
            EXPECT_NEAR(rgb_data[row][base], expected_y, 20);
            EXPECT_NEAR(rgb_data[row][base + 1], expected_y, 20);
            EXPECT_NEAR(rgb_data[row][base + 2], expected_y, 20);
        }
    }
}

TEST_F(JdmrgextTest, H2V2_LargeInput) {

    int even_height = (height % 2 == 0) ? height : height - 1;

    h2v2_merged_upsample(output_width, ycc_input, 0, rgb_data);

    for (int row = 0; row < even_height; row++) {
        for (int col = 0; col < output_width * 3; col++) {
            EXPECT_GE(rgb_data[row][col], 0);
            EXPECT_LE(rgb_data[row][col], 255);
        }
    }
}

TEST_F(JdmrgextTest, ChromaEffectTest) {

    unsigned char y_neutral[8] = {128, 128, 128, 128, 128, 128, 128, 128};
    unsigned char cb_vary[4] = {64, 128, 192, 255};
    unsigned char cr_neutral[4] = {128, 128, 128, 128};
    unsigned char *y_ptr = y_neutral;
    unsigned char *cb_ptr = cb_vary;
    unsigned char *cr_ptr = cr_neutral;
    unsigned char **chroma_input[3] = {&y_ptr, &cb_ptr, &cr_ptr};
    unsigned char rgb_out[24];
    unsigned char *chroma_output = rgb_out;

    h2v1_merged_upsample(8, chroma_input, 0, &chroma_output);

    EXPECT_LT(rgb_out[2], rgb_out[8]);
    EXPECT_LT(rgb_out[8], rgb_out[14]);
    EXPECT_LT(rgb_out[14], rgb_out[20]);
}

TEST_F(JdmrgextTest, LuminanceEffectTest) {

    unsigned char y_vary[8] = {0, 85, 170, 255, 0, 85, 170, 255};
    unsigned char cb_neutral[4] = {128, 128, 128, 128};
    unsigned char cr_neutral[4] = {128, 128, 128, 128};
    unsigned char *y_ptr = y_vary;
    unsigned char *cb_ptr = cb_neutral;
    unsigned char *cr_ptr = cr_neutral;
    unsigned char **luma_input[3] = {&y_ptr, &cb_ptr, &cr_ptr};
    unsigned char rgb_out[24];
    unsigned char *luma_output = rgb_out;

    h2v1_merged_upsample(8, luma_input, 0, &luma_output);

    for (int component = 0; component < 3; component++) {
        EXPECT_LT(rgb_out[component], rgb_out[component + 3]);
        EXPECT_LT(rgb_out[component + 3], rgb_out[component + 6]);
        EXPECT_LT(rgb_out[component + 6], rgb_out[component + 9]);
    }
}

TEST_F(JdmrgextTest, ReproducibilityTest) {

    unsigned char rgb1[16 * 3], rgb2[16 * 3];
    unsigned char *output1 = rgb1, *output2 = rgb2;

    h2v1_merged_upsample(output_width, ycc_input, 0, &output1);
    h2v1_merged_upsample(output_width, ycc_input, 0, &output2);

    for (int i = 0; i < output_width * 3; i++) {
        EXPECT_EQ(rgb1[i], rgb2[i]);
    }
}
