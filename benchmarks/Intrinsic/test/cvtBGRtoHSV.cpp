#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <cmath>

extern int cvtBGRtoHSV(int start, int end, const unsigned char * src, size_t src_step,
                       unsigned char * dst, size_t dst_step, int width, int scn,
                       bool swapBlue, bool isFullRange, bool isHSV);

class CvtBGRtoHSVTest : public ::testing::Test {
protected:
    void SetUp() override {

        width = 16;
        height = 16;
    }

    void TearDown() override {}

    int width, height;

    void validateHSVRange(const std::vector<unsigned char>& hsv, bool isFullRange) {
        int hmax = isFullRange ? 255 : 179;
        for (size_t i = 0; i < hsv.size(); i += 3) {
            EXPECT_LE(hsv[i], hmax) << "H value out of range at position " << i/3;
            EXPECT_LE(hsv[i+1], 255) << "S value out of range at position " << i/3;
            EXPECT_LE(hsv[i+2], 255) << "V value out of range at position " << i/3;
        }
    }
};

TEST_F(CvtBGRtoHSVTest, BGR3_Basic) {
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;
            if (j < width/3) {

                src[idx] = 0; src[idx+1] = 0; src[idx+2] = 255;
            } else if (j < 2*width/3) {

                src[idx] = 0; src[idx+1] = 255; src[idx+2] = 0;
            } else {

                src[idx] = 255; src[idx+1] = 0; src[idx+2] = 0;
            }
        }
    }

    int result = cvtBGRtoHSV(0, height, src.data(), src_step, dst.data(), dst_step,
                             width, scn, false, false, true);

    EXPECT_EQ(result, 0);

    validateHSVRange(dst, false);

    EXPECT_NEAR(dst[0], 0, 3);
    EXPECT_EQ(dst[1], 255);
    EXPECT_EQ(dst[2], 255);

    size_t green_idx = (width/3) * 3;
    EXPECT_NEAR(dst[green_idx], 60, 3);
    EXPECT_EQ(dst[green_idx+1], 255);
    EXPECT_EQ(dst[green_idx+2], 255);

    size_t blue_idx = (2*width/3) * 3;
    EXPECT_NEAR(dst[blue_idx], 120, 3);
    EXPECT_EQ(dst[blue_idx+1], 255);
    EXPECT_EQ(dst[blue_idx+2], 255);
}

TEST_F(CvtBGRtoHSVTest, BGRX4_Basic) {
    const int scn = 4;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;
            if (j < width/3) {

                src[idx] = 0; src[idx+1] = 0; src[idx+2] = 255; src[idx+3] = 255;
            } else if (j < 2*width/3) {

                src[idx] = 0; src[idx+1] = 255; src[idx+2] = 0; src[idx+3] = 128;
            } else {

                src[idx] = 255; src[idx+1] = 0; src[idx+2] = 0; src[idx+3] = 0;
            }
        }
    }

    int result = cvtBGRtoHSV(0, height, src.data(), src_step, dst.data(), dst_step,
                             width, scn, false, false, true);

    EXPECT_EQ(result, 0);

    validateHSVRange(dst, false);

    EXPECT_NEAR(dst[0], 0, 3);
    EXPECT_EQ(dst[1], 255);
    EXPECT_EQ(dst[2], 255);
}

TEST_F(CvtBGRtoHSVTest, FullRangeHue) {
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src = {0, 0, 255};
    std::vector<unsigned char> dst_180(3);
    std::vector<unsigned char> dst_256(3);

    int result1 = cvtBGRtoHSV(0, 1, src.data(), src_step, dst_180.data(), dst_step,
                              1, scn, false, false, true);

    int result2 = cvtBGRtoHSV(0, 1, src.data(), src_step, dst_256.data(), dst_step,
                              1, scn, false, true, true);

    EXPECT_EQ(result1, 0);
    EXPECT_EQ(result2, 0);

    EXPECT_NEAR(dst_180[0], 0, 3);
    EXPECT_NEAR(dst_256[0], 0, 3);

    EXPECT_EQ(dst_180[1], dst_256[1]);
    EXPECT_EQ(dst_180[2], dst_256[2]);
}

TEST_F(CvtBGRtoHSVTest, SwapBlueTest) {
    const int scn = 3;
    const size_t src_step = 1 * scn;
    const size_t dst_step = 1 * 3;

    std::vector<unsigned char> src = {255, 0, 0};
    std::vector<unsigned char> dst_bgr(3);
    std::vector<unsigned char> dst_rgb(3);

    int result1 = cvtBGRtoHSV(0, 1, src.data(), src_step, dst_bgr.data(), dst_step,
                              1, scn, false, false, true);

    int result2 = cvtBGRtoHSV(0, 1, src.data(), src_step, dst_rgb.data(), dst_step,
                              1, scn, true, false, true);

    EXPECT_EQ(result1, 0);
    EXPECT_EQ(result2, 0);

    EXPECT_NEAR(dst_bgr[0], 120, 3);
    EXPECT_NEAR(dst_rgb[0], 0, 3);
}

TEST_F(CvtBGRtoHSVTest, WhitePixel) {
    const int scn = 3;
    const size_t src_step = 8 * scn;
    const size_t dst_step = 8 * 3;

    std::vector<unsigned char> src(8 * src_step, 255);
    std::vector<unsigned char> dst(8 * dst_step);

    int result = cvtBGRtoHSV(0, 8, src.data(), src_step, dst.data(), dst_step,
                             8, scn, false, false, true);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < 8 * 3; i += 3) {
        EXPECT_EQ(dst[i+1], 0);
        EXPECT_EQ(dst[i+2], 255);
    }
}

TEST_F(CvtBGRtoHSVTest, BlackPixel) {
    const int scn = 3;
    const size_t src_step = 8 * scn;
    const size_t dst_step = 8 * 3;

    std::vector<unsigned char> src(8 * src_step, 0);
    std::vector<unsigned char> dst(8 * dst_step);

    int result = cvtBGRtoHSV(0, 8, src.data(), src_step, dst.data(), dst_step,
                             8, scn, false, false, true);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < 8 * 3; i += 3) {
        EXPECT_EQ(dst[i+1], 0);
        EXPECT_EQ(dst[i+2], 0);
    }
}

TEST_F(CvtBGRtoHSVTest, GrayPixel) {
    const int scn = 3;
    const size_t src_step = 1 * scn;
    const size_t dst_step = 1 * 3;

    std::vector<unsigned char> src = {128, 128, 128};
    std::vector<unsigned char> dst(3);

    int result = cvtBGRtoHSV(0, 1, src.data(), src_step, dst.data(), dst_step,
                             1, scn, false, false, true);

    EXPECT_EQ(result, 0);

    EXPECT_EQ(dst[1], 0);
    EXPECT_EQ(dst[2], 128);
}

TEST_F(CvtBGRtoHSVTest, InvalidParameters) {
    const int scn = 3;
    const size_t src_step = 10 * scn;
    const size_t dst_step = 10 * 3;

    std::vector<unsigned char> src(100);
    std::vector<unsigned char> dst(100);

    EXPECT_EQ(cvtBGRtoHSV(0, 10, nullptr, src_step, dst.data(), dst_step, 10, scn, false, false, true), 1);
    EXPECT_EQ(cvtBGRtoHSV(0, 10, src.data(), src_step, nullptr, dst_step, 10, scn, false, false, true), 1);

    EXPECT_EQ(cvtBGRtoHSV(-1, 10, src.data(), src_step, dst.data(), dst_step, 10, scn, false, false, true), 1);
    EXPECT_EQ(cvtBGRtoHSV(10, 5, src.data(), src_step, dst.data(), dst_step, 10, scn, false, false, true), 1);

    EXPECT_EQ(cvtBGRtoHSV(0, 10, src.data(), src_step, dst.data(), dst_step, 0, scn, false, false, true), 1);
    EXPECT_EQ(cvtBGRtoHSV(0, 10, src.data(), src_step, dst.data(), dst_step, -1, scn, false, false, true), 1);

}

TEST_F(CvtBGRtoHSVTest, DifferentSizes) {
    std::vector<std::pair<int, int>> sizes = {{1, 1}, {7, 13}, {64, 64}, {100, 50}};

    for (auto [w, h] : sizes) {
        const int scn = 3;
        const size_t src_step = w * scn;
        const size_t dst_step = w * 3;

        std::vector<unsigned char> src(h * src_step);
        std::vector<unsigned char> dst(h * dst_step);

        for (size_t i = 0; i < src.size(); i += 3) {
            src[i] = src[i+1] = src[i+2] = 128;
        }

        int result = cvtBGRtoHSV(0, h, src.data(), src_step, dst.data(), dst_step,
                                 w, scn, false, false, true);

        EXPECT_EQ(result, 0) << "Failed for size " << w << "x" << h;

        validateHSVRange(dst, false);

        for (int i = 0; i < h * w * 3; i += 3) {
            EXPECT_EQ(dst[i+1], 0) << "S should be 0 for gray at size " << w << "x" << h;
            EXPECT_EQ(dst[i+2], 128) << "V should be 128 for gray at size " << w << "x" << h;
        }
    }
}










TEST_F(CvtBGRtoHSVTest, BoundaryValues) {
    const int scn = 3;
    const size_t src_step = 4 * scn;
    const size_t dst_step = 4 * 3;

    std::vector<unsigned char> src = {
        0, 0, 0,
        255, 255, 255,
        1, 0, 0,
        254, 255, 255
    };
    std::vector<unsigned char> dst(4 * 3);

    int result = cvtBGRtoHSV(0, 1, src.data(), src_step, dst.data(), dst_step,
                             4, scn, false, false, true);

    EXPECT_EQ(result, 0);

    validateHSVRange(dst, false);

    EXPECT_EQ(dst[2], 0);

    EXPECT_EQ(dst[3+1], 0);
    EXPECT_EQ(dst[3+2], 255);

    EXPECT_NEAR(dst[6], 120, 5);
    EXPECT_EQ(dst[6+1], 255);
    EXPECT_EQ(dst[6+2], 1);
}

TEST_F(CvtBGRtoHSVTest, LargeImage) {
    const int width = 320, height = 240;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;

            if (j < width / 6) {

                src[idx] = 0; src[idx+1] = 0; src[idx+2] = 200;
            } else if (j < width / 3) {

                src[idx] = 0; src[idx+1] = 200; src[idx+2] = 0;
            } else if (j < width / 2) {

                src[idx] = 200; src[idx+1] = 0; src[idx+2] = 0;
            } else {

                unsigned char val = static_cast<unsigned char>((j * 255) / width);
                src[idx] = val; src[idx+1] = val; src[idx+2] = val;
            }
        }
    }

    int result = cvtBGRtoHSV(0, height, src.data(), src_step, dst.data(), dst_step,
                             width, scn, false, false, true);

    EXPECT_EQ(result, 0);

    validateHSVRange(dst, false);

    EXPECT_NEAR(dst[0], 0, 5);
    EXPECT_GE(dst[1], 200);
    EXPECT_GE(dst[2], 200);

    size_t green_idx = (width / 6) * 3;
    EXPECT_NEAR(dst[green_idx], 60, 5);
    EXPECT_GE(dst[green_idx+1], 200);
    EXPECT_GE(dst[green_idx+2], 200);

    size_t blue_idx = (width / 3) * 3;
    EXPECT_NEAR(dst[blue_idx], 120, 5);
    EXPECT_GE(dst[blue_idx+1], 200);
    EXPECT_GE(dst[blue_idx+2], 200);
}

TEST_F(CvtBGRtoHSVTest, PartialImageProcessing) {
    const int width = 10, height = 10;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src(height * src_step, 128);
    std::vector<unsigned char> dst(height * dst_step, 99);

    int result = cvtBGRtoHSV(2, 7, src.data(), src_step, dst.data(), dst_step,
                             width, scn, false, false, true);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            size_t dst_idx = (i * width + j) * 3;
            if (i >= 2 && i < 7) {

                EXPECT_EQ(dst[dst_idx+1], 0);
                EXPECT_EQ(dst[dst_idx+2], 128);
            } else {

                EXPECT_EQ(dst[dst_idx], 99);
                EXPECT_EQ(dst[dst_idx+1], 99);
                EXPECT_EQ(dst[dst_idx+2], 99);
            }
        }
    }
}
