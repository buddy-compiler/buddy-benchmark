#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstring>

#define CV_HAL_MORPH_ERODE   0
#define CV_HAL_MORPH_DILATE  1
#define CV_HAL_ERROR_OK      0
#define CV_HAL_ERROR_NOT_IMPLEMENTED  1
#define CV_8U                0

extern int morph(int op, int src_type, int dst_type,
                 uint8_t* src_data, size_t src_step,
                 uint8_t* dst_data, size_t dst_step,
                 int width, int height,
                 int roi_width, int roi_height, int roi_x, int roi_y,
                 int roi_width2, int roi_height2, int roi_x2, int roi_y2,
                 int kernel_type, uint8_t* kernel_data, size_t kernel_step,
                 int kernel_width, int kernel_height, int anchor_x, int anchor_y,
                 int borderType, const double borderValue[4], int iterations, bool isSubmatrix);

class MorphTest : public ::testing::Test {
protected:
    void SetUp() override {

        full_kernel = {1, 1, 1,
                       1, 1, 1,
                       1, 1, 1};

        cross_kernel = {0, 1, 0,
                        1, 1, 1,
                        0, 1, 0};

        borderValue[0] = 0.0;
        borderValue[1] = 0.0;
        borderValue[2] = 0.0;
        borderValue[3] = 0.0;
    }

    std::vector<uint8_t> full_kernel;
    std::vector<uint8_t> cross_kernel;
    double borderValue[4];
};

TEST_F(MorphTest, Simple5x5ErodeWithBorderConstantZero) {
    std::vector<uint8_t> src = {10, 20, 30, 40, 50,
                                60, 70, 80, 90, 100,
                                110, 120, 130, 140, 150,
                                160, 170, 180, 190, 200,
                                210, 220, 230, 240, 250};
    std::vector<uint8_t> dst(25);
    std::vector<uint8_t> expected = {0, 0, 0, 0, 0,
                                     0, 10, 20, 30, 0,
                                     0, 60, 70, 80, 0,
                                     0, 110, 120, 130, 0,
                                     0, 0, 0, 0, 0};

    int ret = morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
                    src.data(), 5, dst.data(), 5, 5, 5,
                    5, 5, 0, 0, 5, 5, 0, 0,
                    CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                    0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, expected);
}

TEST_F(MorphTest, Simple5x5DilateWithBorderConstantZero) {
    std::vector<uint8_t> src = {10, 20, 30, 40, 50,
                                60, 70, 80, 90, 100,
                                110, 120, 130, 140, 150,
                                160, 170, 180, 190, 200,
                                210, 220, 230, 240, 250};
    std::vector<uint8_t> dst(25);
    std::vector<uint8_t> expected = {70,  80,  90, 100, 100,
                                     120, 130, 140, 150, 150,
                                     170, 180, 190, 200, 200,
                                     220, 230, 240, 250, 250,
                                     220, 230, 240, 250, 250};

    int ret = morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
                    src.data(), 5, dst.data(), 5, 5, 5,
                    5, 5, 0, 0, 5, 5, 0, 0,
                    CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                    0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, expected);
}

TEST_F(MorphTest, Simple5x5ErodeWithBorderReplicate) {
    std::vector<uint8_t> src = {10, 20, 30, 40, 50,
                                60, 70, 80, 90, 100,
                                110, 120, 130, 140, 150,
                                160, 170, 180, 190, 200,
                                210, 220, 230, 240, 250};
    std::vector<uint8_t> dst(25);
    std::vector<uint8_t> expected = {10, 10, 20, 30, 40,
                                     10, 10, 20, 30, 40,
                                     60, 60, 70, 80, 90,
                                     110, 110, 120, 130, 140,
                                     160, 160, 170, 180, 190};

    int ret = morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
                    src.data(), 5, dst.data(), 5, 5, 5,
                    5, 5, 0, 0, 5, 5, 0, 0,
                    CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                    4, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, expected);
}

TEST_F(MorphTest, Simple5x5DilateWithBorderReplicate) {
    std::vector<uint8_t> src = {10, 20, 30, 40, 50,
                                60, 70, 80, 90, 100,
                                110, 120, 130, 140, 150,
                                160, 170, 180, 190, 200,
                                210, 220, 230, 240, 250};
    std::vector<uint8_t> dst(25);
    std::vector<uint8_t> expected = {70, 80, 90, 100, 100,
                                    120, 130, 140, 150, 150,
                                    170, 180, 190, 200, 200,
                                    220, 230, 240, 250, 250,
                                    220, 230, 240, 250, 250};

    int ret = morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
                    src.data(), 5, dst.data(), 5, 5, 5,
                    5, 5, 0, 0, 5, 5, 0, 0,
                    CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                    4, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, expected);
}

TEST_F(MorphTest, CrossKernelErode5x5) {
    std::vector<uint8_t> src = {100, 200, 100, 200, 100,
                                200, 255, 200, 255, 200,
                                100, 200, 255, 200, 100,
                                200, 255, 200, 255, 200,
                                100, 200, 100, 200, 100};
    std::vector<uint8_t> dst(25);
    std::vector<uint8_t> expected = {0, 0, 0, 0, 0,
                                     0, 200, 100, 200, 0,
                                     0, 100, 200, 100, 0,
                                     0, 200, 100, 200, 0,
                                     0, 0, 0, 0, 0};

    int ret = morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
                    src.data(), 5, dst.data(), 5, 5, 5,
                    5, 5, 0, 0, 5, 5, 0, 0,
                    CV_8U, cross_kernel.data(), 3, 3, 3, 1, 1,
                    0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, expected);
}

TEST_F(MorphTest, CrossKernelDilate5x5) {
    std::vector<uint8_t> src = {100, 200, 100, 200, 100,
                                200, 255, 200, 255, 200,
                                100, 200, 255, 200, 100,
                                200, 255, 200, 255, 200,
                                100, 200, 100, 200, 100};
    std::vector<uint8_t> dst(25);
    std::vector<uint8_t> expected = {200, 255, 200, 255, 200,
                                     255, 255, 255, 255, 255,
                                     200, 255, 255, 255, 200,
                                     255, 255, 255, 255, 255,
                                     200, 255, 200, 255, 200};

    int ret = morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
                    src.data(), 5, dst.data(), 5, 5, 5,
                    5, 5, 0, 0, 5, 5, 0, 0,
                    CV_8U, cross_kernel.data(), 3, 3, 3, 1, 1,
                    0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, expected);
}

TEST_F(MorphTest, Large7x7ImageBoundaryTest) {
    std::vector<uint8_t> src = {0,   10,  20,  30,  40,  50,  60,
                                70,  80,  90,  100, 110, 120, 130,
                                140, 150, 160, 170, 180, 190, 200,
                                210, 220, 230, 240, 250, 255, 245,
                                235, 225, 215, 205, 195, 185, 175,
                                165, 155, 145, 135, 125, 115, 105,
                                95,  85,  75,  65,  55,  45,  35};
    std::vector<uint8_t> dst(49);

    std::vector<uint8_t> expected_erode = {
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 10, 20, 30, 40, 0,
        0, 70, 80, 90, 100, 110, 0,
        0, 140, 150, 160, 170, 175, 0,
        0, 145, 135, 125, 115, 105, 0,
        0, 75, 65, 55, 45, 35, 0,
        0, 0, 0, 0, 0, 0, 0};

    int ret = morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
                    src.data(), 7, dst.data(), 7, 7, 7,
                    7, 7, 0, 0, 7, 7, 0, 0,
                    CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                    0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, expected_erode);

    std::fill(dst.begin(), dst.end(), 0);
    std::vector<uint8_t> expected_dilate = {80, 90, 100, 110, 120, 130, 130,
                                            150, 160, 170, 180, 190, 200, 200,
                                            220, 230, 240, 250, 255, 255, 255,
                                            235, 235, 240, 250, 255, 255, 255,
                                            235, 235, 240, 250, 255, 255, 255,
                                            235, 235, 225, 215, 205, 195, 185,
                                            165, 165, 155, 145, 135, 125, 115};

    ret = morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
                src.data(), 7, dst.data(), 7, 7, 7,
                7, 7, 0, 0, 7, 7, 0, 0,
                CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, expected_dilate);
}

TEST_F(MorphTest, UniformImage5x5) {
    const int size = 5 * 5;
    std::vector<uint8_t> src(size, 128);
    std::vector<uint8_t> dst(size);

    int ret = morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
                    src.data(), 5, dst.data(), 5, 5, 5,
                    5, 5, 0, 0, 5, 5, 0, 0,
                    CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                    4, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(dst[i], 128);
    }

    std::fill(dst.begin(), dst.end(), 0);
    ret = morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
                src.data(), 5, dst.data(), 5, 5, 5,
                5, 5, 0, 0, 5, 5, 0, 0,
                CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(dst[i], 128);
    }
}

TEST_F(MorphTest, ExtremeLuminanceValues5x5) {
    std::vector<uint8_t> src = {0,   255, 0,   255, 0,
                                255, 128, 255, 128, 255,
                                0,   255, 128, 255, 0,
                                255, 128, 255, 128, 255,
                                0,   255, 0,   255, 0};
    std::vector<uint8_t> dst(25);

    std::vector<uint8_t> expected_erode = {0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0,
                                           0, 0, 128, 0, 0,
                                           0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0};

    int ret = morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
                    src.data(), 5, dst.data(), 5, 5, 5,
                    5, 5, 0, 0, 5, 5, 0, 0,
                    CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                    0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, expected_erode);

    std::fill(dst.begin(), dst.end(), 0);
    std::vector<uint8_t> expected_dilate = {255, 255, 255, 255, 255,
                                            255, 255, 255, 255, 255,
                                            255, 255, 255, 255, 255,
                                            255, 255, 255, 255, 255,
                                            255, 255, 255, 255, 255};

    ret = morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
                src.data(), 5, dst.data(), 5, 5, 5,
                5, 5, 0, 0, 5, 5, 0, 0,
                CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, expected_dilate);
}

TEST_F(MorphTest, SinglePixelKernel5x5) {

    std::vector<uint8_t> center_kernel = {0, 0, 0,
                                          0, 1, 0,
                                          0, 0, 0};

    std::vector<uint8_t> src = {10, 20, 30, 40, 50,
                                60, 70, 80, 90, 100,
                                110, 120, 130, 140, 150,
                                160, 170, 180, 190, 200,
                                210, 220, 230, 240, 250};
    std::vector<uint8_t> dst(25);

    int ret = morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
                    src.data(), 5, dst.data(), 5, 5, 5,
                    5, 5, 0, 0, 5, 5, 0, 0,
                    CV_8U, center_kernel.data(), 3, 3, 3, 1, 1,
                    4, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, src);

    std::fill(dst.begin(), dst.end(), 0);
    ret = morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
                src.data(), 5, dst.data(), 5, 5, 5,
                5, 5, 0, 0, 5, 5, 0, 0,
                CV_8U, center_kernel.data(), 3, 3, 3, 1, 1,
                0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
    EXPECT_EQ(dst, src);
}

TEST_F(MorphTest, DifferentAnchorPoints5x5) {
    std::vector<uint8_t> src = {10, 20, 30, 40, 50,
                                60, 70, 80, 90, 100,
                                110, 120, 130, 140, 150,
                                160, 170, 180, 190, 200,
                                210, 220, 230, 240, 250};
    std::vector<uint8_t> dst(25);

    int ret = morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
                    src.data(), 5, dst.data(), 5, 5, 5,
                    5, 5, 0, 0, 5, 5, 0, 0,
                    CV_8U, full_kernel.data(), 3, 3, 3, 0, 0,
                    0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);

    std::fill(dst.begin(), dst.end(), 0);
    ret = morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
                src.data(), 5, dst.data(), 5, 5, 5,
                5, 5, 0, 0, 5, 5, 0, 0,
                CV_8U, full_kernel.data(), 3, 3, 3, 2, 2,
                0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
}

TEST_F(MorphTest, ROITest) {
    const int width = 10;
    const int height = 10;
    const int size = width * height;

    std::vector<uint8_t> src(size);
    std::vector<uint8_t> dst(size);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            src[y * width + x] = (x + y) % 256;
        }
    }

    int ret = morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
                    src.data(), width, dst.data(), width, width, height,
                    5, 5, 2, 2, 5, 5, 2, 2,
                    CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                    0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
}

TEST_F(MorphTest, LargeImage) {
    const int width = 100;
    const int height = 100;
    const int size = width * height;

    std::vector<uint8_t> src(size);
    std::vector<uint8_t> dst(size);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            src[y * width + x] = ((x + y) % 2) ? 255 : 0;
        }
    }

    int ret = morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
                    src.data(), width, dst.data(), width, width, height,
                    width, height, 0, 0, width, height, 0, 0,
                    CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                    0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);

    ret = morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
                src.data(), width, dst.data(), width, width, height,
                width, height, 0, 0, width, height, 0, 0,
                CV_8U, full_kernel.data(), 3, 3, 3, 1, 1,
                0, borderValue, 1, false);

    EXPECT_EQ(ret, CV_HAL_ERROR_OK);
}
