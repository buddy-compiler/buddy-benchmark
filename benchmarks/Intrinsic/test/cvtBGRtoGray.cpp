#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <cmath>

extern int cvtBGRtoGray(const unsigned char* src_data, size_t src_step,
                        unsigned char* dst_data, size_t dst_step,
                        int width, int height,
                        int depth, int scn);

class CvtBGRtoGrayTest : public ::testing::Test {
protected:
    void SetUp() override {

        width = 16;
        height = 16;
    }

    void TearDown() override {}

    int width, height;
};

TEST_F(CvtBGRtoGrayTest, BGR3_Basic) {
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);
    std::vector<unsigned char> expected(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;
            if (j < width/3) {

                src[idx] = 255; src[idx+1] = 0; src[idx+2] = 0;
                expected[i * dst_step + j] = 29;
            } else if (j < 2*width/3) {

                src[idx] = 0; src[idx+1] = 255; src[idx+2] = 0;
                expected[i * dst_step + j] = 150;
            } else {

                src[idx] = 0; src[idx+1] = 0; src[idx+2] = 255;
                expected[i * dst_step + j] = 76;
            }
        }
    }

    int result = cvtBGRtoGray(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height * dst_step; ++i) {
        EXPECT_NEAR(dst[i], expected[i], 1) << "Mismatch at position " << i;
    }
}

TEST_F(CvtBGRtoGrayTest, BGRX4_Basic) {
    const int scn = 4;
    const size_t src_step = width * scn;
    const size_t dst_step = width;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);
    std::vector<unsigned char> expected(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;
            if (j < width/3) {

                src[idx] = 255; src[idx+1] = 0; src[idx+2] = 0; src[idx+3] = 255;
                expected[i * dst_step + j] = 29;
            } else if (j < 2*width/3) {

                src[idx] = 0; src[idx+1] = 255; src[idx+2] = 0; src[idx+3] = 255;
                expected[i * dst_step + j] = 150;
            } else {

                src[idx] = 0; src[idx+1] = 0; src[idx+2] = 255; src[idx+3] = 255;
                expected[i * dst_step + j] = 76;
            }
        }
    }

    int result = cvtBGRtoGray(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height * dst_step; ++i) {
        EXPECT_NEAR(dst[i], expected[i], 1) << "Mismatch at position " << i;
    }
}

TEST_F(CvtBGRtoGrayTest, WhitePixel) {
    const int width = 8, height = 8;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width;

    std::vector<unsigned char> src(height * src_step, 255);
    std::vector<unsigned char> dst(height * dst_step);

    int result = cvtBGRtoGray(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height * dst_step; ++i) {
        EXPECT_EQ(dst[i], 255);
    }
}

TEST_F(CvtBGRtoGrayTest, BlackPixel) {
    const int width = 8, height = 8;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width;

    std::vector<unsigned char> src(height * src_step, 0);
    std::vector<unsigned char> dst(height * dst_step);

    int result = cvtBGRtoGray(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height * dst_step; ++i) {
        EXPECT_EQ(dst[i], 0);
    }
}

TEST_F(CvtBGRtoGrayTest, InvalidParameters) {
    std::vector<unsigned char> src(100);
    std::vector<unsigned char> dst(100);

    EXPECT_EQ(cvtBGRtoGray(src.data(), 10, dst.data(), 10, 10, 10, 1, 3), 1);

    EXPECT_EQ(cvtBGRtoGray(src.data(), 10, dst.data(), 10, 10, 10, 0, 2), 1);
    EXPECT_EQ(cvtBGRtoGray(src.data(), 10, dst.data(), 10, 10, 10, 0, 5), 1);
}

TEST_F(CvtBGRtoGrayTest, DifferentSizes) {
    std::vector<std::pair<int, int>> sizes = {{1, 1}, {7, 13}, {64, 64}, {100, 50}};

    for (auto [w, h] : sizes) {
        const int scn = 3;
        const size_t src_step = w * scn;
        const size_t dst_step = w;

        std::vector<unsigned char> src(h * src_step, 128);
        std::vector<unsigned char> dst(h * dst_step);

        int result = cvtBGRtoGray(src.data(), src_step,
                                  dst.data(), dst_step,
                                  w, h, 0, scn);

        EXPECT_EQ(result, 0) << "Failed for size " << w << "x" << h;

        for (int i = 0; i < h * dst_step; ++i) {
            EXPECT_EQ(dst[i], 128) << "Failed for size " << w << "x" << h << " at position " << i;
        }
    }
}

TEST_F(CvtBGRtoGrayTest, SinglePixel) {
    const int width = 1, height = 1;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width;

    std::vector<unsigned char> src = {100, 150, 200};
    std::vector<unsigned char> dst(1);

    int result = cvtBGRtoGray(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    EXPECT_EQ(dst[0], 159);
}

TEST_F(CvtBGRtoGrayTest, BoundaryValues) {
    const int width = 4, height = 1;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width;

    std::vector<unsigned char> src = {
        0, 0, 0,
        255, 255, 255,
        128, 128, 128,
        255, 0, 0
    };
    std::vector<unsigned char> dst(width);
    std::vector<unsigned char> expected = {0, 255, 128, 29};

    int result = cvtBGRtoGray(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < width; ++i) {
        EXPECT_NEAR(dst[i], expected[i], 1) << "Failed at pixel " << i;
    }
}

TEST_F(CvtBGRtoGrayTest, BGRX4_IgnoreAlpha) {
    const int width = 2, height = 1;
    const int scn = 4;
    const size_t src_step = width * scn;
    const size_t dst_step = width;

    std::vector<unsigned char> src = {
        100, 150, 200, 0,
        100, 150, 200, 255
    };
    std::vector<unsigned char> dst(width);

    int result = cvtBGRtoGray(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    EXPECT_EQ(dst[0], dst[1]);
    EXPECT_EQ(dst[0], 159);
}

TEST_F(CvtBGRtoGrayTest, LargeImage) {
    const int width = 640, height = 480;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;
            unsigned char val = static_cast<unsigned char>((i + j) % 256);
            src[idx] = val;
            src[idx+1] = val;
            src[idx+2] = val;
        }
    }

    int result = cvtBGRtoGray(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; i += 100) {
        for (int j = 0; j < width; j += 100) {
            unsigned char expected = static_cast<unsigned char>((i + j) % 256);
            EXPECT_EQ(dst[i * dst_step + j], expected)
                << "Failed at position (" << i << "," << j << ")";
        }
    }
}
