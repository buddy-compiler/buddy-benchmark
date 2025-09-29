#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <cmath>

extern int cvtGraytoBGR(const unsigned char* src_data, size_t src_step,
                        unsigned char* dst_data, size_t dst_step,
                        int width, int height,
                        int depth, int scn);

class CvtGraytoBGRTest : public ::testing::Test {
protected:
    void SetUp() override {

        width = 16;
        height = 16;
    }

    void TearDown() override {}

    int width, height;
};

TEST_F(CvtGraytoBGRTest, Gray2BGR3_Basic) {
    const int scn = 3;
    const size_t src_step = width;
    const size_t dst_step = width * scn;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            unsigned char gray = static_cast<unsigned char>((i * width + j) % 256);
            src[i * src_step + j] = gray;
        }
    }

    int result = cvtGraytoBGR(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            unsigned char expected = src[i * src_step + j];
            int dst_idx = i * dst_step + j * scn;

            EXPECT_EQ(dst[dst_idx], expected) << "B channel mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+1], expected) << "G channel mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+2], expected) << "R channel mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(CvtGraytoBGRTest, Gray2BGRX4_Basic) {
    const int scn = 4;
    const size_t src_step = width;
    const size_t dst_step = width * scn;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            unsigned char gray = static_cast<unsigned char>((i * width + j) % 256);
            src[i * src_step + j] = gray;
        }
    }

    int result = cvtGraytoBGR(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            unsigned char expected = src[i * src_step + j];
            int dst_idx = i * dst_step + j * scn;

            EXPECT_EQ(dst[dst_idx], expected) << "B channel mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+1], expected) << "G channel mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+2], expected) << "R channel mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+3], 255) << "Alpha channel mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(CvtGraytoBGRTest, BlackPixel) {
    const int width = 8, height = 8;
    const int scn = 3;
    const size_t src_step = width;
    const size_t dst_step = width * scn;

    std::vector<unsigned char> src(height * src_step, 0);
    std::vector<unsigned char> dst(height * dst_step);

    int result = cvtGraytoBGR(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height * dst_step; ++i) {
        EXPECT_EQ(dst[i], 0);
    }
}

TEST_F(CvtGraytoBGRTest, WhitePixel) {
    const int width = 8, height = 8;
    const int scn = 3;
    const size_t src_step = width;
    const size_t dst_step = width * scn;

    std::vector<unsigned char> src(height * src_step, 255);
    std::vector<unsigned char> dst(height * dst_step);

    int result = cvtGraytoBGR(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height * dst_step; ++i) {
        EXPECT_EQ(dst[i], 255);
    }
}

TEST_F(CvtGraytoBGRTest, MiddleGray) {
    const int width = 8, height = 8;
    const int scn = 4;
    const size_t src_step = width;
    const size_t dst_step = width * scn;

    std::vector<unsigned char> src(height * src_step, 128);
    std::vector<unsigned char> dst(height * dst_step);

    int result = cvtGraytoBGR(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height * width; ++i) {
        int base = i * scn;
        EXPECT_EQ(dst[base], 128);
        EXPECT_EQ(dst[base+1], 128);
        EXPECT_EQ(dst[base+2], 128);
        EXPECT_EQ(dst[base+3], 255);
    }
}

TEST_F(CvtGraytoBGRTest, InvalidParameters) {
    std::vector<unsigned char> src(100);
    std::vector<unsigned char> dst(300);

    EXPECT_EQ(cvtGraytoBGR(src.data(), 10, dst.data(), 30, 10, 10, 1, 3), 1);

    EXPECT_EQ(cvtGraytoBGR(src.data(), 10, dst.data(), 20, 10, 10, 0, 2), 1);
    EXPECT_EQ(cvtGraytoBGR(src.data(), 10, dst.data(), 50, 10, 10, 0, 5), 1);
}

TEST_F(CvtGraytoBGRTest, DifferentSizes) {
    std::vector<std::pair<int, int>> sizes = {{1, 1}, {7, 13}, {64, 64}, {100, 50}};

    for (auto [w, h] : sizes) {
        const int scn = 3;
        const size_t src_step = w;
        const size_t dst_step = w * scn;

        std::vector<unsigned char> src(h * src_step, 200);
        std::vector<unsigned char> dst(h * dst_step);

        int result = cvtGraytoBGR(src.data(), src_step,
                                  dst.data(), dst_step,
                                  w, h, 0, scn);

        EXPECT_EQ(result, 0) << "Failed for size " << w << "x" << h;

        for (int i = 0; i < h * w; ++i) {
            int base = i * scn;
            EXPECT_EQ(dst[base], 200) << "B channel failed for size " << w << "x" << h << " at pixel " << i;
            EXPECT_EQ(dst[base+1], 200) << "G channel failed for size " << w << "x" << h << " at pixel " << i;
            EXPECT_EQ(dst[base+2], 200) << "R channel failed for size " << w << "x" << h << " at pixel " << i;
        }
    }
}

TEST_F(CvtGraytoBGRTest, SinglePixel) {
    const int width = 1, height = 1;
    const int scn = 3;
    const size_t src_step = width;
    const size_t dst_step = width * scn;

    std::vector<unsigned char> src = {100};
    std::vector<unsigned char> dst(scn);

    int result = cvtGraytoBGR(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    EXPECT_EQ(dst[0], 100);
    EXPECT_EQ(dst[1], 100);
    EXPECT_EQ(dst[2], 100);
}

TEST_F(CvtGraytoBGRTest, BoundaryValues) {
    const int width = 4, height = 1;
    const int scn = 3;
    const size_t src_step = width;
    const size_t dst_step = width * scn;

    std::vector<unsigned char> src = {0, 255, 128, 1};
    std::vector<unsigned char> dst(dst_step);

    int result = cvtGraytoBGR(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < width; ++i) {
        unsigned char expected = src[i];
        int base = i * scn;

        EXPECT_EQ(dst[base], expected) << "B channel failed at pixel " << i;
        EXPECT_EQ(dst[base+1], expected) << "G channel failed at pixel " << i;
        EXPECT_EQ(dst[base+2], expected) << "R channel failed at pixel " << i;
    }
}

TEST_F(CvtGraytoBGRTest, BGRX4_AlphaChannel) {
    const int width = 3, height = 1;
    const int scn = 4;
    const size_t src_step = width;
    const size_t dst_step = width * scn;

    std::vector<unsigned char> src = {0, 128, 255};
    std::vector<unsigned char> dst(dst_step);

    int result = cvtGraytoBGR(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < width; ++i) {
        unsigned char expected = src[i];
        int base = i * scn;

        EXPECT_EQ(dst[base], expected) << "B channel failed at pixel " << i;
        EXPECT_EQ(dst[base+1], expected) << "G channel failed at pixel " << i;
        EXPECT_EQ(dst[base+2], expected) << "R channel failed at pixel " << i;
        EXPECT_EQ(dst[base+3], 255) << "Alpha channel should be 255 at pixel " << i;
    }
}

TEST_F(CvtGraytoBGRTest, LargeImage) {
    const int width = 640, height = 480;
    const int scn = 3;
    const size_t src_step = width;
    const size_t dst_step = width * scn;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            unsigned char val = static_cast<unsigned char>((i + j) % 256);
            src[i * src_step + j] = val;
        }
    }

    int result = cvtGraytoBGR(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; i += 100) {
        for (int j = 0; j < width; j += 100) {
            unsigned char expected = static_cast<unsigned char>((i + j) % 256);
            int base = i * dst_step + j * scn;

            EXPECT_EQ(dst[base], expected) << "B channel failed at (" << i << "," << j << ")";
            EXPECT_EQ(dst[base+1], expected) << "G channel failed at (" << i << "," << j << ")";
            EXPECT_EQ(dst[base+2], expected) << "R channel failed at (" << i << "," << j << ")";
        }
    }
}

TEST_F(CvtGraytoBGRTest, CustomStride) {
    const int width = 10, height = 5;
    const int scn = 3;
    const size_t src_step = width + 2;
    const size_t dst_step = width * scn + 6;

    std::vector<unsigned char> src(height * src_step, 0);
    std::vector<unsigned char> dst(height * dst_step, 0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            src[i * src_step + j] = static_cast<unsigned char>(50 + j);
        }
    }

    int result = cvtGraytoBGR(src.data(), src_step,
                              dst.data(), dst_step,
                              width, height, 0, scn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            unsigned char expected = static_cast<unsigned char>(50 + j);
            int base = i * dst_step + j * scn;

            EXPECT_EQ(dst[base], expected) << "B channel failed at (" << i << "," << j << ")";
            EXPECT_EQ(dst[base+1], expected) << "G channel failed at (" << i << "," << j << ")";
            EXPECT_EQ(dst[base+2], expected) << "R channel failed at (" << i << "," << j << ")";
        }
    }
}
