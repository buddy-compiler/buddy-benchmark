#include <gtest/gtest.h>
#include <vector>
#include <cstring>

extern void rgb_ycc_convert(unsigned int image_width, unsigned char **input_buf,
                            unsigned char ***output_buf, unsigned int output_row, int num_rows);

class JccolextTest : public ::testing::Test {
protected:
    void SetUp() override {

        width = 8;
        height = 4;

        for (int row = 0; row < height; row++) {
            rgb_data[row] = new unsigned char[width * 3];
        }

        for (int plane = 0; plane < 3; plane++) {
            for (int row = 0; row < height; row++) {
                ycc_planes[plane][row] = new unsigned char[width];
            }
        }

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int base = col * 3;
                rgb_data[row][base + 0] = (unsigned char)(row * 64 + col * 8);
                rgb_data[row][base + 1] = (unsigned char)(row * 32 + col * 16);
                rgb_data[row][base + 2] = (unsigned char)(row * 16 + col * 32);
            }
        }
    }

    void TearDown() override {

        for (int row = 0; row < height; row++) {
            delete[] rgb_data[row];
        }

        for (int plane = 0; plane < 3; plane++) {
            for (int row = 0; row < height; row++) {
                delete[] ycc_planes[plane][row];
            }
        }
    }

    int width, height;
    unsigned char *rgb_data[4];
    unsigned char *ycc_planes[3][4];
    unsigned char **ycc_output[3] = {ycc_planes[0], ycc_planes[1], ycc_planes[2]};
};

TEST_F(JccolextTest, HandlesZeroWidth) {

    rgb_ycc_convert(0, rgb_data, ycc_output, 0, 1);
    SUCCEED();
}

TEST_F(JccolextTest, HandlesSinglePixel) {

    unsigned char white_rgb[3] = {255, 255, 255};
    unsigned char *white_row = white_rgb;
    unsigned char y_out[1], cb_out[1], cr_out[1];
    unsigned char *y_row = y_out;
    unsigned char *cb_row = cb_out;
    unsigned char *cr_row = cr_out;
    unsigned char **white_output[3] = {&y_row, &cb_row, &cr_row};

    rgb_ycc_convert(1, &white_row, white_output, 0, 1);

    EXPECT_EQ(y_out[0], 255);
    EXPECT_NEAR(cb_out[0], 128, 2);
    EXPECT_NEAR(cr_out[0], 128, 2);

    unsigned char black_rgb[3] = {0, 0, 0};
    unsigned char *black_row = black_rgb;
    unsigned char **black_output[3] = {&y_row, &cb_row, &cr_row};

    rgb_ycc_convert(1, &black_row, black_output, 0, 1);

    EXPECT_EQ(y_out[0], 0);
    EXPECT_NEAR(cb_out[0], 128, 2);
    EXPECT_NEAR(cr_out[0], 128, 2);
}

TEST_F(JccolextTest, HandlesPureColors) {

    unsigned char red_rgb[3] = {255, 0, 0};
    unsigned char *red_row = red_rgb;
    unsigned char y_out[1], cb_out[1], cr_out[1];
    unsigned char *y_row = y_out;
    unsigned char *cb_row = cb_out;
    unsigned char *cr_row = cr_out;
    unsigned char **red_output[3] = {&y_row, &cb_row, &cr_row};

    rgb_ycc_convert(1, &red_row, red_output, 0, 1);

    EXPECT_GT(cr_out[0], 180);
    EXPECT_LT(cb_out[0], 110);

    unsigned char green_rgb[3] = {0, 255, 0};
    unsigned char *green_row = green_rgb;
    unsigned char **green_output[3] = {&y_row, &cb_row, &cr_row};

    rgb_ycc_convert(1, &green_row, green_output, 0, 1);

    EXPECT_GT(y_out[0], 140);
    EXPECT_LT(y_out[0], 160);

    unsigned char blue_rgb[3] = {0, 0, 255};
    unsigned char *blue_row = blue_rgb;
    unsigned char **blue_output[3] = {&y_row, &cb_row, &cr_row};

    rgb_ycc_convert(1, &blue_row, blue_output, 0, 1);

    EXPECT_GT(cb_out[0], 200);
    EXPECT_LT(cr_out[0], 110);
}

TEST_F(JccolextTest, HandlesMultipleRows) {

    rgb_ycc_convert(width, rgb_data, ycc_output, 0, height);

    for (int plane = 0; plane < 3; plane++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                EXPECT_GE(ycc_planes[plane][row][col], 0);
                EXPECT_LE(ycc_planes[plane][row][col], 255);
            }
        }
    }

    bool different_found = false;
    for (int row = 1; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (ycc_planes[0][0][0] != ycc_planes[0][row][col]) {
                different_found = true;
                break;
            }
        }
    }
    EXPECT_TRUE(different_found) << "Expected different outputs for different inputs";
}
