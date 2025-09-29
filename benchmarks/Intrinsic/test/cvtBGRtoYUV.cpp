#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <cmath>

extern void cvtBGRtoYUV(const unsigned char* src_data, size_t src_step,
                        unsigned char* dst_data, size_t dst_step,
                        int width, int height,
                        int depth, int scn);

class CvtBGRtoYUVTest : public ::testing::Test {
protected:
    void SetUp() override {

        width = 16;
        height = 16;
    }

    void TearDown() override {}

    int width, height;

    void reference_cvt_bgr_to_yuv(const unsigned char* src, unsigned char* dst,
                                  int w, int h, int scn) {

        const int yuv_shift = 14;
        const int R2Y = 4899, G2Y = 9617, B2Y = 1868;
        const int YCRI = 11682, YCBI = 9241;

        for (int i = 0; i < h; i++) {
            const unsigned char* src_row = src + i * w * scn;
            unsigned char* dst_row = dst + i * w * 3;

            for (int j = 0; j < w; j++) {
                int b = src_row[j * scn];
                int g = src_row[j * scn + 1];
                int r = src_row[j * scn + 2];

                int Y = (b * B2Y + g * G2Y + r * R2Y + (1 << (yuv_shift - 1))) >> yuv_shift;
                int Cb = ((b - Y) * YCBI + (128 << yuv_shift) + (1 << (yuv_shift - 1))) >> yuv_shift;
                int Cr = ((r - Y) * YCRI + (128 << yuv_shift) + (1 << (yuv_shift - 1))) >> yuv_shift;

                Y = std::max(0, std::min(255, Y));
                Cb = std::max(0, std::min(255, Cb));
                Cr = std::max(0, std::min(255, Cr));

                dst_row[j * 3] = (unsigned char)Y;
                dst_row[j * 3 + 1] = (unsigned char)Cr;
                dst_row[j * 3 + 2] = (unsigned char)Cb;
            }
        }
    }
};

TEST_F(CvtBGRtoYUVTest, BGR3_Basic) {
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);
    std::vector<unsigned char> expected(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;
            if (j < width/3) {

                src[idx] = 255; src[idx+1] = 0; src[idx+2] = 0;
            } else if (j < 2*width/3) {

                src[idx] = 0; src[idx+1] = 255; src[idx+2] = 0;
            } else {

                src[idx] = 0; src[idx+1] = 0; src[idx+2] = 255;
            }
        }
    }

    reference_cvt_bgr_to_yuv(src.data(), expected.data(), width, height, scn);

    cvtBGRtoYUV(src.data(), src_step,
                dst.data(), dst_step,
                width, height, 0, scn);

    for (int i = 0; i < height * dst_step; ++i) {
        EXPECT_NEAR(dst[i], expected[i], 2) << "Mismatch at position " << i;
    }
}

TEST_F(CvtBGRtoYUVTest, BGRX4_Basic) {
    const int scn = 4;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);
    std::vector<unsigned char> expected(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;
            if (j < width/3) {

                src[idx] = 255; src[idx+1] = 0; src[idx+2] = 0; src[idx+3] = 255;
            } else if (j < 2*width/3) {

                src[idx] = 0; src[idx+1] = 255; src[idx+2] = 0; src[idx+3] = 255;
            } else {

                src[idx] = 0; src[idx+1] = 0; src[idx+2] = 255; src[idx+3] = 255;
            }
        }
    }

    reference_cvt_bgr_to_yuv(src.data(), expected.data(), width, height, scn);

    cvtBGRtoYUV(src.data(), src_step,
                dst.data(), dst_step,
                width, height, 0, scn);

    for (int i = 0; i < height * dst_step; ++i) {
        EXPECT_NEAR(dst[i], expected[i], 2) << "Mismatch at position " << i;
    }
}

TEST_F(CvtBGRtoYUVTest, WhitePixel) {
    const int width = 8, height = 8;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src(height * src_step, 255);
    std::vector<unsigned char> dst(height * dst_step);
    std::vector<unsigned char> expected(height * dst_step);

    reference_cvt_bgr_to_yuv(src.data(), expected.data(), width, height, scn);

    cvtBGRtoYUV(src.data(), src_step,
                dst.data(), dst_step,
                width, height, 0, scn);

    for (int i = 0; i < height * dst_step; ++i) {
        EXPECT_NEAR(dst[i], expected[i], 2);
    }

    for (int i = 0; i < height * width; ++i) {
        EXPECT_NEAR(dst[i * 3], 255, 2);
        EXPECT_NEAR(dst[i * 3 + 1], 128, 2);
        EXPECT_NEAR(dst[i * 3 + 2], 128, 2);
    }
}

TEST_F(CvtBGRtoYUVTest, BlackPixel) {
    const int width = 8, height = 8;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src(height * src_step, 0);
    std::vector<unsigned char> dst(height * dst_step);

    cvtBGRtoYUV(src.data(), src_step,
                dst.data(), dst_step,
                width, height, 0, scn);

    for (int i = 0; i < height * width; ++i) {
        EXPECT_NEAR(dst[i * 3], 0, 2);
        EXPECT_NEAR(dst[i * 3 + 1], 128, 2);
        EXPECT_NEAR(dst[i * 3 + 2], 128, 2);
    }
}

TEST_F(CvtBGRtoYUVTest, PureColors) {
    const int width = 3, height = 1;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src = {
        0, 0, 255,
        0, 255, 0,
        255, 0, 0
    };
    std::vector<unsigned char> dst(height * dst_step);
    std::vector<unsigned char> expected(height * dst_step);

    reference_cvt_bgr_to_yuv(src.data(), expected.data(), width, height, scn);

    cvtBGRtoYUV(src.data(), src_step,
                dst.data(), dst_step,
                width, height, 0, scn);

    for (int i = 0; i < width * 3; ++i) {
        EXPECT_NEAR(dst[i], expected[i], 2) << "Failed at position " << i;
    }
}

TEST_F(CvtBGRtoYUVTest, DifferentSizes) {
    std::vector<std::pair<int, int>> sizes = {{1, 1}, {7, 13}, {64, 64}, {100, 50}};

    for (auto [w, h] : sizes) {
        const int scn = 3;
        const size_t src_step = w * scn;
        const size_t dst_step = w * 3;

        std::vector<unsigned char> src(h * src_step, 128);
        std::vector<unsigned char> dst(h * dst_step);
        std::vector<unsigned char> expected(h * dst_step);

        reference_cvt_bgr_to_yuv(src.data(), expected.data(), w, h, scn);

        cvtBGRtoYUV(src.data(), src_step,
                    dst.data(), dst_step,
                    w, h, 0, scn);

        for (int i = 0; i < h * dst_step; ++i) {
            EXPECT_NEAR(dst[i], expected[i], 2)
                << "Failed for size " << w << "x" << h << " at position " << i;
        }
    }
}

TEST_F(CvtBGRtoYUVTest, SinglePixel) {
    const int width = 1, height = 1;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src = {100, 150, 200};
    std::vector<unsigned char> dst(3);
    std::vector<unsigned char> expected(3);

    reference_cvt_bgr_to_yuv(src.data(), expected.data(), width, height, scn);

    cvtBGRtoYUV(src.data(), src_step,
                dst.data(), dst_step,
                width, height, 0, scn);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(dst[i], expected[i], 2) << "Failed at component " << i;
    }
}

TEST_F(CvtBGRtoYUVTest, BoundaryValues) {
    const int width = 4, height = 1;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src = {
        0, 0, 0,
        255, 255, 255,
        128, 128, 128,
        255, 0, 0
    };
    std::vector<unsigned char> dst(width * 3);
    std::vector<unsigned char> expected(width * 3);

    reference_cvt_bgr_to_yuv(src.data(), expected.data(), width, height, scn);

    cvtBGRtoYUV(src.data(), src_step,
                dst.data(), dst_step,
                width, height, 0, scn);

    for (int i = 0; i < width * 3; ++i) {
        EXPECT_NEAR(dst[i], expected[i], 2) << "Failed at position " << i;
    }
}

TEST_F(CvtBGRtoYUVTest, BGRX4_IgnoreAlpha) {
    const int width = 2, height = 1;
    const int scn = 4;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src = {
        100, 150, 200, 0,
        100, 150, 200, 255
    };
    std::vector<unsigned char> dst(width * 3);

    cvtBGRtoYUV(src.data(), src_step,
                dst.data(), dst_step,
                width, height, 0, scn);

    EXPECT_EQ(dst[0], dst[3]);
    EXPECT_EQ(dst[1], dst[4]);
    EXPECT_EQ(dst[2], dst[5]);
}

TEST_F(CvtBGRtoYUVTest, LargeImage) {
    const int width = 640, height = 480;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 3;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);
    std::vector<unsigned char> expected(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;
            unsigned char val = static_cast<unsigned char>((i + j) % 256);
            src[idx] = val;
            src[idx+1] = val;
            src[idx+2] = val;
        }
    }

    reference_cvt_bgr_to_yuv(src.data(), expected.data(), width, height, scn);

    cvtBGRtoYUV(src.data(), src_step,
                dst.data(), dst_step,
                width, height, 0, scn);

    for (int i = 0; i < height; i += 100) {
        for (int j = 0; j < width; j += 100) {
            int dst_idx = (i * width + j) * 3;
            int exp_idx = dst_idx;
            EXPECT_NEAR(dst[dst_idx], expected[exp_idx], 2)
                << "Y component failed at position (" << i << "," << j << ")";
            EXPECT_NEAR(dst[dst_idx+1], expected[exp_idx+1], 2)
                << "Cr component failed at position (" << i << "," << j << ")";
            EXPECT_NEAR(dst[dst_idx+2], expected[exp_idx+2], 2)
                << "Cb component failed at position (" << i << "," << j << ")";
        }
    }
}
