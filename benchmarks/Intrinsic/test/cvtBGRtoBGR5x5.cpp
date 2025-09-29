#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <cmath>

extern int cvtBGRtoBGR5x5(const unsigned char* src_data, size_t src_step,
                          unsigned char* dst_data, size_t dst_step,
                          int width, int height,
                          int scn, bool swapBlue, int greenBits);

class CvtBGRtoBGR5x5Test : public ::testing::Test {
protected:
    void SetUp() override {

        width = 16;
        height = 16;
    }

    void TearDown() override {}

    int width, height;

    void decodeBGR5x5(uint16_t encoded, uint8_t& r, uint8_t& g, uint8_t& b, int greenBits) {
        if (greenBits == 6) {

            b = static_cast<uint8_t>((encoded & 0x001F) << 3);
            g = static_cast<uint8_t>(((encoded & 0x07E0) >> 3) & 0xFC);
            r = static_cast<uint8_t>(((encoded & 0xF800) >> 8) & 0xF8);
        } else {

            b = static_cast<uint8_t>((encoded & 0x001F) << 3);
            g = static_cast<uint8_t>(((encoded & 0x03E0) >> 2) & 0xF8);
            r = static_cast<uint8_t>(((encoded & 0x7C00) >> 7) & 0xF8);
        }
    }
};

TEST_F(CvtBGRtoBGR5x5Test, BGR3_to_BGR565_Basic) {
    const int scn = 3;
    const int greenBits = 6;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 2;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

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

    int result = cvtBGRtoBGR5x5(src.data(), src_step,
                                dst.data(), dst_step,
                                width, height, scn, false, greenBits);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int src_idx = i * src_step + j * scn;
            int dst_idx = i * dst_step + j * 2;

            uint16_t encoded = *((uint16_t*)(dst.data() + dst_idx));
            uint8_t decoded_r, decoded_g, decoded_b;
            decodeBGR5x5(encoded, decoded_r, decoded_g, decoded_b, greenBits);

            int tolerance = 8;

            EXPECT_NEAR(decoded_b, src[src_idx], tolerance)
                << "Blue mismatch at (" << i << "," << j << ")";
            EXPECT_NEAR(decoded_g, src[src_idx+1], tolerance)
                << "Green mismatch at (" << i << "," << j << ")";
            EXPECT_NEAR(decoded_r, src[src_idx+2], tolerance)
                << "Red mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(CvtBGRtoBGR5x5Test, BGR3_to_BGR555_Basic) {
    const int scn = 3;
    const int greenBits = 5;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 2;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

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

    int result = cvtBGRtoBGR5x5(src.data(), src_step,
                                dst.data(), dst_step,
                                width, height, scn, false, greenBits);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int src_idx = i * src_step + j * scn;
            int dst_idx = i * dst_step + j * 2;

            uint16_t encoded = *((uint16_t*)(dst.data() + dst_idx));
            uint8_t decoded_r, decoded_g, decoded_b;
            decodeBGR5x5(encoded, decoded_r, decoded_g, decoded_b, greenBits);

            int tolerance = 8;

            EXPECT_NEAR(decoded_b, src[src_idx], tolerance)
                << "Blue mismatch at (" << i << "," << j << ")";
            EXPECT_NEAR(decoded_g, src[src_idx+1], tolerance)
                << "Green mismatch at (" << i << "," << j << ")";
            EXPECT_NEAR(decoded_r, src[src_idx+2], tolerance)
                << "Red mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(CvtBGRtoBGR5x5Test, BGRX4_to_BGR565_Basic) {
    const int scn = 4;
    const int greenBits = 6;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 2;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;
            if (j < width/3) {

                src[idx] = 255; src[idx+1] = 0; src[idx+2] = 0; src[idx+3] = 128;
            } else if (j < 2*width/3) {

                src[idx] = 0; src[idx+1] = 255; src[idx+2] = 0; src[idx+3] = 64;
            } else {

                src[idx] = 0; src[idx+1] = 0; src[idx+2] = 255; src[idx+3] = 255;
            }
        }
    }

    int result = cvtBGRtoBGR5x5(src.data(), src_step,
                                dst.data(), dst_step,
                                width, height, scn, false, greenBits);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int src_idx = i * src_step + j * scn;
            int dst_idx = i * dst_step + j * 2;

            uint16_t encoded = *((uint16_t*)(dst.data() + dst_idx));
            uint8_t decoded_r, decoded_g, decoded_b;
            decodeBGR5x5(encoded, decoded_r, decoded_g, decoded_b, greenBits);

            int tolerance = 8;

            EXPECT_NEAR(decoded_b, src[src_idx], tolerance);
            EXPECT_NEAR(decoded_g, src[src_idx+1], tolerance);
            EXPECT_NEAR(decoded_r, src[src_idx+2], tolerance);

        }
    }
}

TEST_F(CvtBGRtoBGR5x5Test, SwapBlue_Functionality) {
    const int width = 4, height = 1;
    const int scn = 3;
    const int greenBits = 6;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 2;

    std::vector<unsigned char> src = {
        100, 150, 200,
        50, 100, 150,
        0, 128, 255,
        255, 255, 255
    };

    std::vector<unsigned char> dst_normal(height * dst_step);
    std::vector<unsigned char> dst_swapped(height * dst_step);

    int result1 = cvtBGRtoBGR5x5(src.data(), src_step,
                                 dst_normal.data(), dst_step,
                                 width, height, scn, false, greenBits);

    int result2 = cvtBGRtoBGR5x5(src.data(), src_step,
                                 dst_swapped.data(), dst_step,
                                 width, height, scn, true, greenBits);

    EXPECT_EQ(result1, 0);
    EXPECT_EQ(result2, 0);

    for (int j = 0; j < width; ++j) {
        int src_idx = j * scn;
        int dst_idx = j * 2;

        uint16_t encoded_normal = *((uint16_t*)(dst_normal.data() + dst_idx));
        uint16_t encoded_swapped = *((uint16_t*)(dst_swapped.data() + dst_idx));

        uint8_t normal_r, normal_g, normal_b;
        uint8_t swapped_r, swapped_g, swapped_b;

        decodeBGR5x5(encoded_normal, normal_r, normal_g, normal_b, greenBits);
        decodeBGR5x5(encoded_swapped, swapped_r, swapped_g, swapped_b, greenBits);

        int tolerance = 8;

        EXPECT_NEAR(normal_b, src[src_idx], tolerance);
        EXPECT_NEAR(normal_g, src[src_idx+1], tolerance);
        EXPECT_NEAR(normal_r, src[src_idx+2], tolerance);

        EXPECT_NEAR(swapped_b, src[src_idx+2], tolerance);
        EXPECT_NEAR(swapped_g, src[src_idx+1], tolerance);
        EXPECT_NEAR(swapped_r, src[src_idx], tolerance);
    }
}

TEST_F(CvtBGRtoBGR5x5Test, SinglePixel) {
    const int width = 1, height = 1;
    const int scn = 3;
    const int greenBits = 6;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 2;

    std::vector<unsigned char> src = {100, 150, 200};
    std::vector<unsigned char> dst(2);

    int result = cvtBGRtoBGR5x5(src.data(), src_step,
                                dst.data(), dst_step,
                                width, height, scn, false, greenBits);

    EXPECT_EQ(result, 0);

    uint16_t encoded = *((uint16_t*)dst.data());
    uint8_t decoded_r, decoded_g, decoded_b;
    decodeBGR5x5(encoded, decoded_r, decoded_g, decoded_b, greenBits);

    int tolerance = 8;
    EXPECT_NEAR(decoded_b, 100, tolerance);
    EXPECT_NEAR(decoded_g, 150, tolerance);
    EXPECT_NEAR(decoded_r, 200, tolerance);
}

TEST_F(CvtBGRtoBGR5x5Test, BoundaryValues) {
    const int width = 4, height = 1;
    const int scn = 3;
    const int greenBits = 6;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 2;

    std::vector<unsigned char> src = {
        0, 0, 0,
        255, 255, 255,
        128, 128, 128,
        255, 0, 0
    };
    std::vector<unsigned char> dst(dst_step);

    int result = cvtBGRtoBGR5x5(src.data(), src_step,
                                dst.data(), dst_step,
                                width, height, scn, false, greenBits);

    EXPECT_EQ(result, 0);

    for (int j = 0; j < width; ++j) {
        int src_idx = j * scn;
        int dst_idx = j * 2;

        uint16_t encoded = *((uint16_t*)(dst.data() + dst_idx));
        uint8_t decoded_r, decoded_g, decoded_b;
        decodeBGR5x5(encoded, decoded_r, decoded_g, decoded_b, greenBits);

        int tolerance = 8;

        EXPECT_NEAR(decoded_b, src[src_idx], tolerance) << "Failed at pixel " << j;
        EXPECT_NEAR(decoded_g, src[src_idx+1], tolerance) << "Failed at pixel " << j;
        EXPECT_NEAR(decoded_r, src[src_idx+2], tolerance) << "Failed at pixel " << j;
    }
}

TEST_F(CvtBGRtoBGR5x5Test, InvalidParameters) {
    std::vector<unsigned char> src(100);
    std::vector<unsigned char> dst(100);

    EXPECT_EQ(cvtBGRtoBGR5x5(src.data(), 10, dst.data(), 10, 10, 10, 2, false, 6), 1);
    EXPECT_EQ(cvtBGRtoBGR5x5(src.data(), 10, dst.data(), 10, 10, 10, 5, false, 6), 1);

    EXPECT_EQ(cvtBGRtoBGR5x5(src.data(), 10, dst.data(), 10, 10, 10, 3, false, 4), 1);
    EXPECT_EQ(cvtBGRtoBGR5x5(src.data(), 10, dst.data(), 10, 10, 10, 3, false, 7), 1);

    EXPECT_EQ(cvtBGRtoBGR5x5(nullptr, 10, dst.data(), 10, 10, 10, 3, false, 6), 1);
    EXPECT_EQ(cvtBGRtoBGR5x5(src.data(), 10, nullptr, 10, 10, 10, 3, false, 6), 1);

    EXPECT_EQ(cvtBGRtoBGR5x5(src.data(), 10, dst.data(), 10, 0, 10, 3, false, 6), 1);
    EXPECT_EQ(cvtBGRtoBGR5x5(src.data(), 10, dst.data(), 10, 10, 0, 3, false, 6), 1);
}

TEST_F(CvtBGRtoBGR5x5Test, DifferentSizes) {
    std::vector<std::pair<int, int>> sizes = {{1, 1}, {7, 13}, {64, 64}, {100, 50}};

    for (auto [w, h] : sizes) {
        const int scn = 3;
        const int greenBits = 6;
        const size_t src_step = w * scn;
        const size_t dst_step = w * 2;

        std::vector<unsigned char> src(h * src_step, 128);
        std::vector<unsigned char> dst(h * dst_step);

        int result = cvtBGRtoBGR5x5(src.data(), src_step,
                                    dst.data(), dst_step,
                                    w, h, scn, false, greenBits);

        EXPECT_EQ(result, 0) << "Failed for size " << w << "x" << h;

        for (int i = 0; i < h; i += std::max(1, h/4)) {
            for (int j = 0; j < w; j += std::max(1, w/4)) {
                int dst_idx = i * dst_step + j * 2;
                uint16_t encoded = *((uint16_t*)(dst.data() + dst_idx));
                uint8_t decoded_r, decoded_g, decoded_b;
                decodeBGR5x5(encoded, decoded_r, decoded_g, decoded_b, greenBits);

                int tolerance = 8;
                EXPECT_NEAR(decoded_b, 128, tolerance)
                    << "Failed for size " << w << "x" << h << " at position (" << i << "," << j << ")";
                EXPECT_NEAR(decoded_g, 128, tolerance)
                    << "Failed for size " << w << "x" << h << " at position (" << i << "," << j << ")";
                EXPECT_NEAR(decoded_r, 128, tolerance)
                    << "Failed for size " << w << "x" << h << " at position (" << i << "," << j << ")";
            }
        }
    }
}

TEST_F(CvtBGRtoBGR5x5Test, Compare_BGR555_BGR565) {
    const int width = 8, height = 1;
    const int scn = 3;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 2;

    std::vector<unsigned char> src(src_step);
    for (int j = 0; j < width; ++j) {
        int idx = j * scn;
        unsigned char val = static_cast<unsigned char>(j * 32);
        src[idx] = val;
        src[idx+1] = val;
        src[idx+2] = val;
    }

    std::vector<unsigned char> dst555(dst_step);
    std::vector<unsigned char> dst565(dst_step);

    int result555 = cvtBGRtoBGR5x5(src.data(), src_step,
                                   dst555.data(), dst_step,
                                   width, 1, scn, false, 5);

    int result565 = cvtBGRtoBGR5x5(src.data(), src_step,
                                   dst565.data(), dst_step,
                                   width, 1, scn, false, 6);

    EXPECT_EQ(result555, 0);
    EXPECT_EQ(result565, 0);

    for (int j = 0; j < width; ++j) {
        int src_idx = j * scn;
        int dst_idx = j * 2;

        uint16_t encoded555 = *((uint16_t*)(dst555.data() + dst_idx));
        uint16_t encoded565 = *((uint16_t*)(dst565.data() + dst_idx));

        uint8_t r555, g555, b555;
        uint8_t r565, g565, b565;

        decodeBGR5x5(encoded555, r555, g555, b555, 5);
        decodeBGR5x5(encoded565, r565, g565, b565, 6);

        int green_error_555 = std::abs(g555 - src[src_idx+1]);
        int green_error_565 = std::abs(g565 - src[src_idx+1]);

        if (src[src_idx+1] > 0) {
            EXPECT_LE(green_error_565, green_error_555 + 1)
                << "BGR565 should have better or equal green precision at pixel " << j
                << " (src_g=" << (int)src[src_idx+1] << ", g555=" << (int)g555
                << ", g565=" << (int)g565 << ")";
        }
    }
}

TEST_F(CvtBGRtoBGR5x5Test, LargeImage) {
    const int width = 640, height = 480;
    const int scn = 3;
    const int greenBits = 6;
    const size_t src_step = width * scn;
    const size_t dst_step = width * 2;

    std::vector<unsigned char> src(height * src_step);
    std::vector<unsigned char> dst(height * dst_step);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * src_step + j * scn;
            unsigned char val_b = static_cast<unsigned char>((i + j) % 256);
            unsigned char val_g = static_cast<unsigned char>((i * 2 + j) % 256);
            unsigned char val_r = static_cast<unsigned char>((i + j * 2) % 256);
            src[idx] = val_b;
            src[idx+1] = val_g;
            src[idx+2] = val_r;
        }
    }

    int result = cvtBGRtoBGR5x5(src.data(), src_step,
                                dst.data(), dst_step,
                                width, height, scn, false, greenBits);

    EXPECT_EQ(result, 0);

    for (int i = 0; i < height; i += 100) {
        for (int j = 0; j < width; j += 100) {
            int src_idx = i * src_step + j * scn;
            int dst_idx = i * dst_step + j * 2;

            uint16_t encoded = *((uint16_t*)(dst.data() + dst_idx));
            uint8_t decoded_r, decoded_g, decoded_b;
            decodeBGR5x5(encoded, decoded_r, decoded_g, decoded_b, greenBits);

            int tolerance = 8;

            EXPECT_NEAR(decoded_b, src[src_idx], tolerance)
                << "Failed at position (" << i << "," << j << ")";
            EXPECT_NEAR(decoded_g, src[src_idx+1], tolerance)
                << "Failed at position (" << i << "," << j << ")";
            EXPECT_NEAR(decoded_r, src[src_idx+2], tolerance)
                << "Failed at position (" << i << "," << j << ")";
        }
    }
}
