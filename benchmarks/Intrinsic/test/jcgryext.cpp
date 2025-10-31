#include <gtest/gtest.h>
#include <vector>
#include <cstring>

extern void rgb_gray_convert(unsigned int image_width, unsigned char **input_buf,
                             unsigned char ***output_buf, unsigned int output_row, int num_rows);

class JcgryextTest : public ::testing::Test {
protected:
    void SetUp() override {

        width = 8;
        height = 4;

        for (int row = 0; row < height; row++) {
            rgb_data[row] = new unsigned char[width * 3];
        }

        for (int row = 0; row < height; row++) {
            gray_plane[row] = new unsigned char[width];
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
            delete[] gray_plane[row];
        }
    }

    int width, height;
    unsigned char *rgb_data[4];
    unsigned char *gray_plane[4];
    unsigned char **gray_output[1] = {gray_plane};
};

TEST_F(JcgryextTest, HandlesEmptyInput) {

    rgb_gray_convert(width, rgb_data, gray_output, 0, 0);
    SUCCEED();
}

TEST_F(JcgryextTest, HandlesZeroWidth) {

    rgb_gray_convert(0, rgb_data, gray_output, 0, 1);
    SUCCEED();
}

TEST_F(JcgryextTest, HandlesSinglePixel) {

    unsigned char white_rgb[3] = {255, 255, 255};
    unsigned char *white_row = white_rgb;
    unsigned char gray_out[1];
    unsigned char *gray_row = gray_out;
    unsigned char **white_output[1] = {&gray_row};

    rgb_gray_convert(1, &white_row, white_output, 0, 1);

    EXPECT_EQ(gray_out[0], 255);

    unsigned char black_rgb[3] = {0, 0, 0};
    unsigned char *black_row = black_rgb;
    unsigned char **black_output[1] = {&gray_row};

    rgb_gray_convert(1, &black_row, black_output, 0, 1);

    EXPECT_EQ(gray_out[0], 0);
}

TEST_F(JcgryextTest, HandlesPureColors) {
    unsigned char gray_out[1];
    unsigned char *gray_row = gray_out;
    unsigned char **color_output[1] = {&gray_row};

    unsigned char red_rgb[3] = {255, 0, 0};
    unsigned char *red_row = red_rgb;
    rgb_gray_convert(1, &red_row, color_output, 0, 1);
    unsigned char red_gray = gray_out[0];

    unsigned char green_rgb[3] = {0, 255, 0};
    unsigned char *green_row = green_rgb;
    rgb_gray_convert(1, &green_row, color_output, 0, 1);
    unsigned char green_gray = gray_out[0];

    unsigned char blue_rgb[3] = {0, 0, 255};
    unsigned char *blue_row = blue_rgb;
    rgb_gray_convert(1, &blue_row, color_output, 0, 1);
    unsigned char blue_gray = gray_out[0];

    EXPECT_GT(green_gray, red_gray);
    EXPECT_GT(green_gray, blue_gray);

    EXPECT_GT(red_gray, blue_gray);

    EXPECT_GT(red_gray, 50);
    EXPECT_GT(green_gray, 100);
    EXPECT_GT(blue_gray, 20);
}

TEST_F(JcgryextTest, HandlesGrayscaleInput) {

    unsigned char gray_out[3];
    unsigned char *gray_rows[3] = {&gray_out[0], &gray_out[1], &gray_out[2]};
    unsigned char **gray_output_test[1] = {gray_rows};

    unsigned char rgb1[3] = {100, 100, 100};
    unsigned char rgb2[3] = {150, 150, 150};
    unsigned char rgb3[3] = {50, 50, 50};

    unsigned char *rows[3] = {rgb1, rgb2, rgb3};

    rgb_gray_convert(1, rows, gray_output_test, 0, 3);

    EXPECT_NEAR(gray_out[0], 100, 2);
    EXPECT_NEAR(gray_out[1], 150, 2);
    EXPECT_NEAR(gray_out[2], 50, 2);

    EXPECT_LT(gray_out[2], gray_out[0]);
    EXPECT_LT(gray_out[0], gray_out[1]);
}

TEST_F(JcgryextTest, HandlesMultipleRows) {

    rgb_gray_convert(width, rgb_data, gray_output, 0, height);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            EXPECT_GE(gray_plane[row][col], 0);
            EXPECT_LE(gray_plane[row][col], 255);
        }
    }

    bool different_found = false;
    for (int row = 1; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (gray_plane[0][0] != gray_plane[row][col]) {
                different_found = true;
                break;
            }
        }
    }
    EXPECT_TRUE(different_found) << "Expected different outputs for different inputs";
}

TEST_F(JcgryextTest, LuminanceFormulaTest) {

    unsigned char test_rgb[3] = {100, 150, 200};
    unsigned char *test_row = test_rgb;
    unsigned char gray_out[1];
    unsigned char *gray_row = gray_out;
    unsigned char **test_output[1] = {&gray_row};

    rgb_gray_convert(1, &test_row, test_output, 0, 1);

    EXPECT_NEAR(gray_out[0], 141, 3);
}
