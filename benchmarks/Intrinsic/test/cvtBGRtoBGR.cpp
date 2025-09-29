#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <cmath>

extern int cvtBGRtoBGR(const unsigned char* src_data, size_t src_step,
                       unsigned char* dst_data, size_t dst_step,
                       int width, int height,
                       int depth, int scn, int dcn);

class CvtBGRtoBGRTest : public ::testing::Test {
protected:
    void SetUp() override {

        width = 16;
        height = 16;
    }

    void TearDown() override {}

    int width, height;
};

TEST_F(CvtBGRtoBGRTest, BGR3toBGR4_Basic) {
    const int scn = 3, dcn = 4;
    const size_t src_step = width * scn;
    const size_t dst_step = width * dcn;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int src_idx = i * src_step + j * scn;
            if (j < width/3) {

                src[src_idx] = 255; src[src_idx+1] = 0; src[src_idx+2] = 0;
            } else if (j < 2*width/3) {

                src[src_idx] = 0; src[src_idx+1] = 255; src[src_idx+2] = 0;
            } else {

                src[src_idx] = 0; src[src_idx+1] = 0; src[src_idx+2] = 255;
            }
        }
    }

    int result = cvtBGRtoBGR(src.data(), src_step,
                             dst.data(), dst_step,
                             width, height, 0, scn, dcn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int src_idx = i * src_step + j * scn;
            int dst_idx = i * dst_step + j * dcn;

            EXPECT_EQ(dst[dst_idx], src[src_idx]) << "B channel mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+1], src[src_idx+1]) << "G channel mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+2], src[src_idx+2]) << "R channel mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+3], 255) << "Alpha channel should be 255 at (" << i << "," << j << ")";
        }
    }
}

TEST_F(CvtBGRtoBGRTest, BGR4toBGR3_Basic) {
    const int scn = 4, dcn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * dcn;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int src_idx = i * src_step + j * scn;
            if (j < width/3) {

                src[src_idx] = 255; src[src_idx+1] = 0; src[src_idx+2] = 0; src[src_idx+3] = 100;
            } else if (j < 2*width/3) {

                src[src_idx] = 0; src[src_idx+1] = 255; src[src_idx+2] = 0; src[src_idx+3] = 200;
            } else {

                src[src_idx] = 0; src[src_idx+1] = 0; src[src_idx+2] = 255; src[src_idx+3] = 50;
            }
        }
    }

    int result = cvtBGRtoBGR(src.data(), src_step,
                             dst.data(), dst_step,
                             width, height, 0, scn, dcn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int src_idx = i * src_step + j * scn;
            int dst_idx = i * dst_step + j * dcn;

            EXPECT_EQ(dst[dst_idx], src[src_idx]) << "B channel mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+1], src[src_idx+1]) << "G channel mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+2], src[src_idx+2]) << "R channel mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(CvtBGRtoBGRTest, BGR3toBGR3_Copy) {
    const int scn = 3, dcn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * dcn;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height * src_step; ++i) {
        src[i] = static_cast<unsigned char>(i % 256);
    }

    int result = cvtBGRtoBGR(src.data(), src_step,
                             dst.data(), dst_step,
                             width, height, 0, scn, dcn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height * src_step; ++i) {
        EXPECT_EQ(dst[i], src[i]) << "Mismatch at position " << i;
    }
}

TEST_F(CvtBGRtoBGRTest, BGR4toBGR4_Copy) {
    const int scn = 4, dcn = 4;
    const size_t src_step = width * scn;
    const size_t dst_step = width * dcn;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height * src_step; ++i) {
        src[i] = static_cast<unsigned char>((i * 17 + 42) % 256);
    }

    int result = cvtBGRtoBGR(src.data(), src_step,
                             dst.data(), dst_step,
                             width, height, 0, scn, dcn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height * src_step; ++i) {
        EXPECT_EQ(dst[i], src[i]) << "Mismatch at position " << i;
    }
}

TEST_F(CvtBGRtoBGRTest, SpecificPixelValues) {
    const int width = 4, height = 1;
    const int scn = 3, dcn = 4;
    const size_t src_step = width * scn;
    const size_t dst_step = width * dcn;

    std::vector<unsigned char> src = {
        0, 0, 0,
        255, 255, 255,
        128, 64, 192,
        50, 100, 200
    };
    std::vector<unsigned char> dst(width * dcn);

    int result = cvtBGRtoBGR(src.data(), src_step,
                             dst.data(), dst_step,
                             width, height, 0, scn, dcn);

    EXPECT_EQ(result, 0);

    std::vector<unsigned char> expected = {
        0, 0, 0, 255,
        255, 255, 255, 255,
        128, 64, 192, 255,
        50, 100, 200, 255
    };

    for (int i = 0; i < width * dcn; ++i) {
        EXPECT_EQ(dst[i], expected[i]) << "Failed at position " << i;
    }
}

TEST_F(CvtBGRtoBGRTest, InvalidParameters) {
    std::vector<unsigned char> src(100);
    std::vector<unsigned char> dst(100);

    EXPECT_EQ(cvtBGRtoBGR(src.data(), 10, dst.data(), 10, 10, 10, 1, 3, 4), 1);

    EXPECT_EQ(cvtBGRtoBGR(src.data(), 10, dst.data(), 10, 10, 10, 0, 2, 3), 1);
    EXPECT_EQ(cvtBGRtoBGR(src.data(), 10, dst.data(), 10, 10, 10, 0, 3, 2), 1);
    EXPECT_EQ(cvtBGRtoBGR(src.data(), 10, dst.data(), 10, 10, 10, 0, 5, 4), 1);
}

TEST_F(CvtBGRtoBGRTest, DifferentSizes) {
    std::vector<std::pair<int, int>> sizes = {{1, 1}, {7, 13}, {64, 64}, {100, 50}};

    for (auto [w, h] : sizes) {
        const int scn = 3, dcn = 4;
        const size_t src_step = w * scn;
        const size_t dst_step = w * dcn;

        std::vector<unsigned char> src(h * src_step);
        std::vector<unsigned char> dst(h * dst_step);

        for (int i = 0; i < h * src_step; ++i) {
            src[i] = static_cast<unsigned char>((i * 31) % 256);
        }

        int result = cvtBGRtoBGR(src.data(), src_step,
                                 dst.data(), dst_step,
                                 w, h, 0, scn, dcn);

        EXPECT_EQ(result, 0) << "Failed for size " << w << "x" << h;

        for (int i = 0; i < h; i += h/4 + 1) {
            for (int j = 0; j < w; j += w/4 + 1) {
                int src_idx = i * src_step + j * scn;
                int dst_idx = i * dst_step + j * dcn;

                EXPECT_EQ(dst[dst_idx], src[src_idx]) << "Size " << w << "x" << h << " B mismatch at (" << i << "," << j << ")";
                EXPECT_EQ(dst[dst_idx+1], src[src_idx+1]) << "Size " << w << "x" << h << " G mismatch at (" << i << "," << j << ")";
                EXPECT_EQ(dst[dst_idx+2], src[src_idx+2]) << "Size " << w << "x" << h << " R mismatch at (" << i << "," << j << ")";
                EXPECT_EQ(dst[dst_idx+3], 255) << "Size " << w << "x" << h << " A should be 255 at (" << i << "," << j << ")";
            }
        }
    }
}

TEST_F(CvtBGRtoBGRTest, SinglePixel) {

    {
        const int width = 1, height = 1;
        const int scn = 3, dcn = 4;
        const size_t src_step = width * scn;
        const size_t dst_step = width * dcn;

        std::vector<unsigned char> src = {123, 45, 67};
        std::vector<unsigned char> dst(dcn);

        int result = cvtBGRtoBGR(src.data(), src_step,
                                 dst.data(), dst_step,
                                 width, height, 0, scn, dcn);

        EXPECT_EQ(result, 0);
        EXPECT_EQ(dst[0], 123);
        EXPECT_EQ(dst[1], 45);
        EXPECT_EQ(dst[2], 67);
        EXPECT_EQ(dst[3], 255);
    }

    {
        const int width = 1, height = 1;
        const int scn = 4, dcn = 3;
        const size_t src_step = width * scn;
        const size_t dst_step = width * dcn;

        std::vector<unsigned char> src = {123, 45, 67, 89};
        std::vector<unsigned char> dst(dcn);

        int result = cvtBGRtoBGR(src.data(), src_step,
                                 dst.data(), dst_step,
                                 width, height, 0, scn, dcn);

        EXPECT_EQ(result, 0);
        EXPECT_EQ(dst[0], 123);
        EXPECT_EQ(dst[1], 45);
        EXPECT_EQ(dst[2], 67);

    }
}

TEST_F(CvtBGRtoBGRTest, BoundaryValues) {
    const int width = 3, height = 1;
    const int scn = 3, dcn = 4;
    const size_t src_step = width * scn;
    const size_t dst_step = width * dcn;

    std::vector<unsigned char> src = {
        0, 0, 0,
        255, 255, 255,
        128, 128, 128
    };
    std::vector<unsigned char> dst(width * dcn);

    int result = cvtBGRtoBGR(src.data(), src_step,
                             dst.data(), dst_step,
                             width, height, 0, scn, dcn);

    EXPECT_EQ(result, 0);

    std::vector<unsigned char> expected = {
        0, 0, 0, 255,
        255, 255, 255, 255,
        128, 128, 128, 255
    };

    for (int i = 0; i < width * dcn; ++i) {
        EXPECT_EQ(dst[i], expected[i]) << "Failed at position " << i;
    }
}

TEST_F(CvtBGRtoBGRTest, AlphaIgnored) {
    const int width = 2, height = 1;
    const int scn = 4, dcn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * dcn;

    std::vector<unsigned char> src = {
        100, 150, 200, 0,
        100, 150, 200, 255
    };
    std::vector<unsigned char> dst(width * dcn);

    int result = cvtBGRtoBGR(src.data(), src_step,
                             dst.data(), dst_step,
                             width, height, 0, scn, dcn);

    EXPECT_EQ(result, 0);

    for (int c = 0; c < dcn; ++c) {
        EXPECT_EQ(dst[c], dst[dcn + c]) << "Channel " << c << " should be same for both pixels";
    }

    EXPECT_EQ(dst[0], 100); EXPECT_EQ(dst[1], 150); EXPECT_EQ(dst[2], 200);
    EXPECT_EQ(dst[3], 100); EXPECT_EQ(dst[4], 150); EXPECT_EQ(dst[5], 200);
}

TEST_F(CvtBGRtoBGRTest, LargeImage) {
    const int width = 640, height = 480;
    const int scn = 3, dcn = 4;
    const size_t src_step = width * scn;
    const size_t dst_step = width * dcn;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;
            src[idx] = static_cast<unsigned char>((i * 10 + j) % 256);
            src[idx+1] = static_cast<unsigned char>((i + j * 5) % 256);
            src[idx+2] = static_cast<unsigned char>((i * 3 + j * 7) % 256);
        }
    }

    int result = cvtBGRtoBGR(src.data(), src_step,
                             dst.data(), dst_step,
                             width, height, 0, scn, dcn);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; i += 100) {
        for (int j = 0; j < width; j += 100) {
            int src_idx = i * src_step + j * scn;
            int dst_idx = i * dst_step + j * dcn;

            EXPECT_EQ(dst[dst_idx], src[src_idx]) << "B failed at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+1], src[src_idx+1]) << "G failed at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+2], src[src_idx+2]) << "R failed at (" << i << "," << j << ")";
            EXPECT_EQ(dst[dst_idx+3], 255) << "A should be 255 at (" << i << "," << j << ")";
        }
    }
}

TEST_F(CvtBGRtoBGRTest, UnalignedWidth) {
    std::vector<int> widths = {1, 7, 15, 17, 31, 33};
    const int height = 8;

    for (int w : widths) {
        const int scn = 3, dcn = 4;
        const size_t src_step = w * scn;
        const size_t dst_step = w * dcn;

        std::vector<unsigned char> src(height * src_step);
        std::vector<unsigned char> dst(height * dst_step);

        for (int i = 0; i < height * src_step; ++i) {
            src[i] = static_cast<unsigned char>((i * 13) % 256);
        }

        int result = cvtBGRtoBGR(src.data(), src_step,
                                 dst.data(), dst_step,
                                 w, height, 0, scn, dcn);

        EXPECT_EQ(result, 0) << "Failed for width " << w;

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < w; ++j) {
                int src_idx = i * src_step + j * scn;
                int dst_idx = i * dst_step + j * dcn;

                EXPECT_EQ(dst[dst_idx], src[src_idx]) << "Width " << w << " B mismatch at (" << i << "," << j << ")";
                EXPECT_EQ(dst[dst_idx+1], src[src_idx+1]) << "Width " << w << " G mismatch at (" << i << "," << j << ")";
                EXPECT_EQ(dst[dst_idx+2], src[src_idx+2]) << "Width " << w << " R mismatch at (" << i << "," << j << ")";
                EXPECT_EQ(dst[dst_idx+3], 255) << "Width " << w << " A should be 255 at (" << i << "," << j << ")";
            }
        }
    }
}
