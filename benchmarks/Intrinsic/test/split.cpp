#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

extern void split8u(const unsigned char* src, unsigned char** dst, int len, int cn);

TEST(SplitTest, HandlesEmptyArray) {
    std::vector<unsigned char> src;
    std::vector<unsigned char> dst0(1);
    unsigned char* dst_ptrs[1] = {dst0.data()};

    split8u(src.data(), dst_ptrs, 0, 1);

    EXPECT_EQ(dst0[0], 0);
}

TEST(SplitTest, HandlesSingleChannel) {
    const int len = 5;
    std::vector<unsigned char> src = {1, 2, 3, 4, 5};
    std::vector<unsigned char> dst0(len);
    unsigned char* dst_ptrs[1] = {dst0.data()};

    split8u(src.data(), dst_ptrs, len, 1);

    for (int i = 0; i < len; ++i) {
        EXPECT_EQ(dst0[i], src[i]);
    }
}

TEST(SplitTest, HandlesTwoChannels) {
    const int len = 4;
    const int cn = 2;
    std::vector<unsigned char> src = {1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<unsigned char> dst0(len), dst1(len);
    unsigned char* dst_ptrs[2] = {dst0.data(), dst1.data()};

    split8u(src.data(), dst_ptrs, len, cn);

    std::vector<unsigned char> expected0 = {1, 3, 5, 7};
    std::vector<unsigned char> expected1 = {2, 4, 6, 8};

    for (int i = 0; i < len; ++i) {
        EXPECT_EQ(dst0[i], expected0[i]);
        EXPECT_EQ(dst1[i], expected1[i]);
    }
}

TEST(SplitTest, HandlesThreeChannels) {
    const int len = 3;
    const int cn = 3;
    std::vector<unsigned char> src = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    std::vector<unsigned char> dst0(len), dst1(len), dst2(len);
    unsigned char* dst_ptrs[3] = {dst0.data(), dst1.data(), dst2.data()};

    split8u(src.data(), dst_ptrs, len, cn);

    std::vector<unsigned char> expected0 = {1, 4, 7};
    std::vector<unsigned char> expected1 = {2, 5, 8};
    std::vector<unsigned char> expected2 = {3, 6, 9};

    for (int i = 0; i < len; ++i) {
        EXPECT_EQ(dst0[i], expected0[i]);
        EXPECT_EQ(dst1[i], expected1[i]);
        EXPECT_EQ(dst2[i], expected2[i]);
    }
}

TEST(SplitTest, HandlesFourChannels) {
    const int len = 2;
    const int cn = 4;
    std::vector<unsigned char> src = {1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<unsigned char> dst0(len), dst1(len), dst2(len), dst3(len);
    unsigned char* dst_ptrs[4] = {dst0.data(), dst1.data(), dst2.data(), dst3.data()};

    split8u(src.data(), dst_ptrs, len, cn);

    std::vector<unsigned char> expected0 = {1, 5};
    std::vector<unsigned char> expected1 = {2, 6};
    std::vector<unsigned char> expected2 = {3, 7};
    std::vector<unsigned char> expected3 = {4, 8};

    for (int i = 0; i < len; ++i) {
        EXPECT_EQ(dst0[i], expected0[i]);
        EXPECT_EQ(dst1[i], expected1[i]);
        EXPECT_EQ(dst2[i], expected2[i]);
        EXPECT_EQ(dst3[i], expected3[i]);
    }
}

TEST(SplitTest, HandlesFiveChannels) {
    const int len = 2;
    const int cn = 5;
    std::vector<unsigned char> src = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::vector<unsigned char> dst0(len), dst1(len), dst2(len), dst3(len), dst4(len);
    unsigned char* dst_ptrs[5] = {dst0.data(), dst1.data(), dst2.data(), dst3.data(), dst4.data()};

    split8u(src.data(), dst_ptrs, len, cn);

    std::vector<unsigned char> expected0 = {1, 6};
    std::vector<unsigned char> expected1 = {2, 7};
    std::vector<unsigned char> expected2 = {3, 8};
    std::vector<unsigned char> expected3 = {4, 9};
    std::vector<unsigned char> expected4 = {5, 10};

    for (int i = 0; i < len; ++i) {
        EXPECT_EQ(dst0[i], expected0[i]);
        EXPECT_EQ(dst1[i], expected1[i]);
        EXPECT_EQ(dst2[i], expected2[i]);
        EXPECT_EQ(dst3[i], expected3[i]);
        EXPECT_EQ(dst4[i], expected4[i]);
    }
}

TEST(SplitTest, HandlesEightChannels) {
    const int len = 2;
    const int cn = 8;
    std::vector<unsigned char> src(len * cn);

    for (int i = 0; i < len * cn; ++i) {
        src[i] = static_cast<unsigned char>(i + 1);
    }

    std::vector<std::vector<unsigned char>> dst_arrays(cn, std::vector<unsigned char>(len));
    std::vector<unsigned char*> dst_ptrs(cn);
    for (int i = 0; i < cn; ++i) {
        dst_ptrs[i] = dst_arrays[i].data();
    }

    split8u(src.data(), dst_ptrs.data(), len, cn);

    for (int ch = 0; ch < cn; ++ch) {
        for (int i = 0; i < len; ++i) {
            unsigned char expected = static_cast<unsigned char>(i * cn + ch + 1);
            EXPECT_EQ(dst_arrays[ch][i], expected);
        }
    }
}

TEST(SplitTest, HandlesSingleElement) {
    const int len = 1;
    const int cn = 3;
    std::vector<unsigned char> src = {10, 20, 30};

    std::vector<unsigned char> dst0(len), dst1(len), dst2(len);
    unsigned char* dst_ptrs[3] = {dst0.data(), dst1.data(), dst2.data()};

    split8u(src.data(), dst_ptrs, len, cn);

    EXPECT_EQ(dst0[0], 10);
    EXPECT_EQ(dst1[0], 20);
    EXPECT_EQ(dst2[0], 30);
}

TEST(SplitTest, HandlesLargeArray) {
    const int len = 1000;
    const int cn = 4;
    std::vector<unsigned char> src(len * cn);

    for (int i = 0; i < len * cn; ++i) {
        src[i] = static_cast<unsigned char>(i % 256);
    }

    std::vector<std::vector<unsigned char>> dst_arrays(cn, std::vector<unsigned char>(len));
    std::vector<unsigned char*> dst_ptrs(cn);
    for (int i = 0; i < cn; ++i) {
        dst_ptrs[i] = dst_arrays[i].data();
    }

    split8u(src.data(), dst_ptrs.data(), len, cn);

    for (int ch = 0; ch < cn; ++ch) {
        for (int i = 0; i < len; ++i) {
            unsigned char expected = static_cast<unsigned char>((i * cn + ch) % 256);
            EXPECT_EQ(dst_arrays[ch][i], expected);
        }
    }
}

TEST(SplitTest, HandlesMaxValues) {
    const int len = 3;
    const int cn = 2;
    std::vector<unsigned char> src = {255, 0, 128, 64, 192, 32};

    std::vector<unsigned char> dst0(len), dst1(len);
    unsigned char* dst_ptrs[2] = {dst0.data(), dst1.data()};

    split8u(src.data(), dst_ptrs, len, cn);

    std::vector<unsigned char> expected0 = {255, 128, 192};
    std::vector<unsigned char> expected1 = {0, 64, 32};

    for (int i = 0; i < len; ++i) {
        EXPECT_EQ(dst0[i], expected0[i]);
        EXPECT_EQ(dst1[i], expected1[i]);
    }
}
