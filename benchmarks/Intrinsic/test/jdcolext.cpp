#include <gtest/gtest.h>
#include <vector>
#include <cstring>

extern void ycc_rgb_convert(unsigned int output_width, unsigned char ***input_buf,
                            unsigned int input_row, unsigned char **output_buf, int num_rows);

class JdcolextTest : public ::testing::Test {
protected:
    void SetUp() override {

        width = 8;
        height = 4;

        for (int plane = 0; plane < 3; plane++) {
            for (int row = 0; row < height; row++) {
                ycc_planes[plane][row] = new unsigned char[width];
            }
        }

        for (int row = 0; row < height; row++) {
            rgb_data[row] = new unsigned char[width * 3];
        }

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                ycc_planes[0][row][col] = (unsigned char)(row * 32 + col * 16);
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

    int width, height;
    unsigned char *ycc_planes[3][4];
    unsigned char **ycc_input[3] = {ycc_planes[0], ycc_planes[1], ycc_planes[2]};
    unsigned char *rgb_data[4];
};

TEST_F(JdcolextTest, HandlesEmptyInput) {

    ycc_rgb_convert(width, ycc_input, 0, rgb_data, 0);
    SUCCEED();
}

TEST_F(JdcolextTest, HandlesZeroWidth) {

    ycc_rgb_convert(0, ycc_input, 0, rgb_data, 1);
    SUCCEED();
}

TEST_F(JdcolextTest, HandlesSinglePixel) {

    unsigned char y_in[1] = {255};
    unsigned char cb_in[1] = {128};
    unsigned char cr_in[1] = {128};
    unsigned char *y_row = y_in;
    unsigned char *cb_row = cb_in;
    unsigned char *cr_row = cr_in;
    unsigned char **white_input[3] = {&y_row, &cb_row, &cr_row};
    unsigned char rgb_out[3];
    unsigned char *rgb_row = rgb_out;

    ycc_rgb_convert(1, white_input, 0, &rgb_row, 1);

    EXPECT_GT(rgb_out[0], 240);
    EXPECT_GT(rgb_out[1], 240);
    EXPECT_GT(rgb_out[2], 240);

    unsigned char y_black[1] = {0};
    unsigned char **black_input[3] = {&y_row, &cb_row, &cr_row};
    y_row = y_black;

    ycc_rgb_convert(1, black_input, 0, &rgb_row, 1);

    EXPECT_LT(rgb_out[0], 15);
    EXPECT_LT(rgb_out[1], 15);
    EXPECT_LT(rgb_out[2], 15);
}

TEST_F(JdcolextTest, HandlesNeutralChroma) {

    unsigned char y_neutral[1] = {128};
    unsigned char cb_neutral[1] = {128};
    unsigned char cr_neutral[1] = {128};
    unsigned char *y_row = y_neutral;
    unsigned char *cb_row = cb_neutral;
    unsigned char *cr_row = cr_neutral;
    unsigned char **neutral_input[3] = {&y_row, &cb_row, &cr_row};
    unsigned char rgb_out[3];
    unsigned char *rgb_row = rgb_out;

    ycc_rgb_convert(1, neutral_input, 0, &rgb_row, 1);

    EXPECT_NEAR(rgb_out[0], 128, 10);
    EXPECT_NEAR(rgb_out[1], 128, 10);
    EXPECT_NEAR(rgb_out[2], 128, 10);

    EXPECT_NEAR(rgb_out[0], rgb_out[1], 5);
    EXPECT_NEAR(rgb_out[1], rgb_out[2], 5);
}

TEST_F(JdcolextTest, HandlesHighCr) {

    unsigned char y_val[1] = {128};
    unsigned char cb_val[1] = {128};
    unsigned char cr_val[1] = {200};
    unsigned char *y_row = y_val;
    unsigned char *cb_row = cb_val;
    unsigned char *cr_row = cr_val;
    unsigned char **red_input[3] = {&y_row, &cb_row, &cr_row};
    unsigned char rgb_out[3];
    unsigned char *rgb_row = rgb_out;

    ycc_rgb_convert(1, red_input, 0, &rgb_row, 1);

    EXPECT_GT(rgb_out[0], rgb_out[1]);
    EXPECT_GT(rgb_out[0], rgb_out[2]);
}

TEST_F(JdcolextTest, HandlesHighCb) {

    unsigned char y_val[1] = {128};
    unsigned char cb_val[1] = {200};
    unsigned char cr_val[1] = {128};
    unsigned char *y_row = y_val;
    unsigned char *cb_row = cb_val;
    unsigned char *cr_row = cr_val;
    unsigned char **blue_input[3] = {&y_row, &cb_row, &cr_row};
    unsigned char rgb_out[3];
    unsigned char *rgb_row = rgb_out;

    ycc_rgb_convert(1, blue_input, 0, &rgb_row, 1);

    EXPECT_GT(rgb_out[2], rgb_out[0]);
    EXPECT_GT(rgb_out[2], rgb_out[1]);
}

TEST_F(JdcolextTest, HandlesMultipleRows) {

    ycc_rgb_convert(width, ycc_input, 0, rgb_data, height);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int base = col * 3;
            EXPECT_GE(rgb_data[row][base + 0], 0);
            EXPECT_LE(rgb_data[row][base + 0], 255);
            EXPECT_GE(rgb_data[row][base + 1], 0);
            EXPECT_LE(rgb_data[row][base + 1], 255);
            EXPECT_GE(rgb_data[row][base + 2], 0);
            EXPECT_LE(rgb_data[row][base + 2], 255);
        }
    }

    bool different_found = false;
    for (int row = 1; row < height; row++) {
        for (int col = 0; col < width * 3; col++) {
            if (rgb_data[0][col] != rgb_data[row][col]) {
                different_found = true;
                break;
            }
        }
    }
    EXPECT_TRUE(different_found) << "Expected different outputs for different inputs";
}

TEST_F(JdcolextTest, YOnlyConversion) {

    unsigned char y1[1] = {50};
    unsigned char y2[1] = {200};
    unsigned char cb_neutral[1] = {128};
    unsigned char cr_neutral[1] = {128};

    unsigned char *cb_row = cb_neutral;
    unsigned char *cr_row = cr_neutral;
    unsigned char *y_row = y1;
    unsigned char **dark_input[3] = {&y_row, &cb_row, &cr_row};
    unsigned char rgb_dark[3];
    unsigned char *rgb_row = rgb_dark;

    ycc_rgb_convert(1, dark_input, 0, &rgb_row, 1);

    y_row = y2;
    unsigned char **bright_input[3] = {&y_row, &cb_row, &cr_row};
    unsigned char rgb_bright[3];
    rgb_row = rgb_bright;

    ycc_rgb_convert(1, bright_input, 0, &rgb_row, 1);

    EXPECT_LT(rgb_dark[0], rgb_bright[0]);
    EXPECT_LT(rgb_dark[1], rgb_bright[1]);
    EXPECT_LT(rgb_dark[2], rgb_bright[2]);
}

TEST_F(JdcolextTest, RangeLimiting) {

    unsigned char y_extreme[1] = {255};
    unsigned char cb_extreme[1] = {255};
    unsigned char cr_extreme[1] = {255};
    unsigned char *y_row = y_extreme;
    unsigned char *cb_row = cb_extreme;
    unsigned char *cr_row = cr_extreme;
    unsigned char **extreme_input[3] = {&y_row, &cb_row, &cr_row};
    unsigned char rgb_out[3];
    unsigned char *rgb_row = rgb_out;

    ycc_rgb_convert(1, extreme_input, 0, &rgb_row, 1);

    EXPECT_GE(rgb_out[0], 0);
    EXPECT_LE(rgb_out[0], 255);
    EXPECT_GE(rgb_out[1], 0);
    EXPECT_LE(rgb_out[1], 255);
    EXPECT_GE(rgb_out[2], 0);
    EXPECT_LE(rgb_out[2], 255);
}
