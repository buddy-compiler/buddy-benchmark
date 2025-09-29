#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

extern void merge8u(const unsigned char** src, unsigned char* dst, int len, int cn);

TEST(MergeTest, HandlesSingleChannel) {
    const int len = 5;
    std::vector<unsigned char> src0 = {1, 2, 3, 4, 5};
    const unsigned char* src_ptrs[1] = {src0.data()};
    std::vector<unsigned char> dst(len);

    merge8u(src_ptrs, dst.data(), len, 1);

    for (int i = 0; i < len; ++i) {
        EXPECT_EQ(dst[i], src0[i]);
    }
}

TEST(MergeTest, HandlesTwoChannels) {
    const int len = 4;
    const int cn = 2;
    std::vector<unsigned char> src0 = {1, 3, 5, 7};
    std::vector<unsigned char> src1 = {2, 4, 6, 8};
    const unsigned char* src_ptrs[2] = {src0.data(), src1.data()};

    std::vector<unsigned char> dst(len * cn);
    merge8u(src_ptrs, dst.data(), len, cn);

    std::vector<unsigned char> expected = {1, 2, 3, 4, 5, 6, 7, 8};

    for (int i = 0; i < len * cn; ++i) {
        EXPECT_EQ(dst[i], expected[i]);
    }
}

TEST(MergeTest, HandlesThreeChannels) {
    const int len = 3;
    const int cn = 3;
    std::vector<unsigned char> src0 = {1, 4, 7};
    std::vector<unsigned char> src1 = {2, 5, 8};
    std::vector<unsigned char> src2 = {3, 6, 9};
    const unsigned char* src_ptrs[3] = {src0.data(), src1.data(), src2.data()};

    std::vector<unsigned char> dst(len * cn);
    merge8u(src_ptrs, dst.data(), len, cn);

    std::vector<unsigned char> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    for (int i = 0; i < len * cn; ++i) {
        EXPECT_EQ(dst[i], expected[i]);
    }
}

TEST(MergeTest, HandlesFourChannels) {
    const int len = 2;
    const int cn = 4;
    std::vector<unsigned char> src0 = {1, 5};
    std::vector<unsigned char> src1 = {2, 6};
    std::vector<unsigned char> src2 = {3, 7};
    std::vector<unsigned char> src3 = {4, 8};
    const unsigned char* src_ptrs[4] = {src0.data(), src1.data(), src2.data(), src3.data()};

    std::vector<unsigned char> dst(len * cn);
    merge8u(src_ptrs, dst.data(), len, cn);

    std::vector<unsigned char> expected = {1, 2, 3, 4, 5, 6, 7, 8};

    for (int i = 0; i < len * cn; ++i) {
        EXPECT_EQ(dst[i], expected[i]);
    }
}

TEST(MergeTest, HandlesFiveChannels) {
    const int len = 2;
    const int cn = 5;
    std::vector<unsigned char> src0 = {1, 6};
    std::vector<unsigned char> src1 = {2, 7};
    std::vector<unsigned char> src2 = {3, 8};
    std::vector<unsigned char> src3 = {4, 9};
    std::vector<unsigned char> src4 = {5, 10};
    const unsigned char* src_ptrs[5] = {src0.data(), src1.data(), src2.data(), src3.data(), src4.data()};

    std::vector<unsigned char> dst(len * cn);
    merge8u(src_ptrs, dst.data(), len, cn);

    std::vector<unsigned char> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (int i = 0; i < len * cn; ++i) {
        EXPECT_EQ(dst[i], expected[i]);
    }
}

TEST(MergeTest, HandlesEightChannels) {
    const int len = 2;
    const int cn = 8;

    std::vector<std::vector<unsigned char>> src_arrays(cn, std::vector<unsigned char>(len));
    std::vector<const unsigned char*> src_ptrs(cn);

    for (int ch = 0; ch < cn; ++ch) {
        for (int i = 0; i < len; ++i) {
            src_arrays[ch][i] = static_cast<unsigned char>(i * cn + ch + 1);
        }
        src_ptrs[ch] = src_arrays[ch].data();
    }

    std::vector<unsigned char> dst(len * cn);
    merge8u(src_ptrs.data(), dst.data(), len, cn);

    for (int i = 0; i < len; ++i) {
        for (int ch = 0; ch < cn; ++ch) {
            unsigned char expected = static_cast<unsigned char>(i * cn + ch + 1);
            EXPECT_EQ(dst[i * cn + ch], expected);
        }
    }
}

TEST(MergeTest, HandlesSingleElement) {
    const int len = 1;
    const int cn = 3;
    std::vector<unsigned char> src0 = {10};
    std::vector<unsigned char> src1 = {20};
    std::vector<unsigned char> src2 = {30};
    const unsigned char* src_ptrs[3] = {src0.data(), src1.data(), src2.data()};

    std::vector<unsigned char> dst(len * cn);
    merge8u(src_ptrs, dst.data(), len, cn);

    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], 20);
    EXPECT_EQ(dst[2], 30);
}

TEST(MergeTest, HandlesLargeArray) {
    const int len = 1000;
    const int cn = 4;

    std::vector<std::vector<unsigned char>> src_arrays(cn, std::vector<unsigned char>(len));
    std::vector<const unsigned char*> src_ptrs(cn);

    for (int ch = 0; ch < cn; ++ch) {
        for (int i = 0; i < len; ++i) {
            src_arrays[ch][i] = static_cast<unsigned char>((i * cn + ch) % 256);
        }
        src_ptrs[ch] = src_arrays[ch].data();
    }

    std::vector<unsigned char> dst(len * cn);
    merge8u(src_ptrs.data(), dst.data(), len, cn);

    for (int i = 0; i < len; ++i) {
        for (int ch = 0; ch < cn; ++ch) {
            unsigned char expected = static_cast<unsigned char>((i * cn + ch) % 256);
            EXPECT_EQ(dst[i * cn + ch], expected);
        }
    }
}

TEST(MergeTest, HandlesMaxValues) {
    const int len = 3;
    const int cn = 2;
    std::vector<unsigned char> src0 = {255, 128, 192};
    std::vector<unsigned char> src1 = {0, 64, 32};
    const unsigned char* src_ptrs[2] = {src0.data(), src1.data()};

    std::vector<unsigned char> dst(len * cn);
    merge8u(src_ptrs, dst.data(), len, cn);

    std::vector<unsigned char> expected = {255, 0, 128, 64, 192, 32};

    for (int i = 0; i < len * cn; ++i) {
        EXPECT_EQ(dst[i], expected[i]);
    }
}

TEST(MergeTest, TestMergeSplitRoundTrip) {

    const int len = 10;
    const int cn = 3;

    std::vector<unsigned char> original(len * cn);
    for (int i = 0; i < len * cn; ++i) {
        original[i] = static_cast<unsigned char>(i % 256);
    }

    std::vector<std::vector<unsigned char>> split_arrays(cn, std::vector<unsigned char>(len));
    std::vector<unsigned char*> split_ptrs(cn);
    for (int i = 0; i < cn; ++i) {
        split_ptrs[i] = split_arrays[i].data();
    }

    for (int i = 0; i < len; ++i) {
        for (int ch = 0; ch < cn; ++ch) {
            split_arrays[ch][i] = original[i * cn + ch];
        }
    }

    std::vector<unsigned char> merged(len * cn);
    std::vector<const unsigned char*> const_split_ptrs(cn);
    for (int i = 0; i < cn; ++i) {
        const_split_ptrs[i] = split_arrays[i].data();
    }

    merge8u(const_split_ptrs.data(), merged.data(), len, cn);

    for (int i = 0; i < len * cn; ++i) {
        EXPECT_EQ(merged[i], original[i]);
    }
}

TEST(MergeTest, HandlesZeroValues) {
    const int len = 4;
    const int cn = 2;
    std::vector<unsigned char> src0 = {0, 0, 0, 0};
    std::vector<unsigned char> src1 = {0, 0, 0, 0};
    const unsigned char* src_ptrs[2] = {src0.data(), src1.data()};

    std::vector<unsigned char> dst(len * cn);
    merge8u(src_ptrs, dst.data(), len, cn);

    for (int i = 0; i < len * cn; ++i) {
        EXPECT_EQ(dst[i], 0);
    }
}
