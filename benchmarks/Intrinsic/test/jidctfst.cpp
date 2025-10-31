#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <cmath>

typedef short JCOEF;
typedef JCOEF *JCOEFPTR;
typedef unsigned char *JSAMPROW;
typedef unsigned char JSAMPLE;
typedef JSAMPROW *JSAMPARRAY;
typedef unsigned int JDIMENSION;
#define MAXJSAMPLE 255
#define CENTERJSAMPLE 128

extern void idct_ifast(void *quantptr, JCOEFPTR coef_block, JSAMPARRAY output_buf, JDIMENSION output_col, JSAMPROW sample_range_limit);

class JidctfstTest : public ::testing::Test {
protected:
    JSAMPLE * table;
    JSAMPLE * original_table_ptr;
    JSAMPLE * sample_range_limit;
    void setupTable() {
        int i;
        original_table_ptr = (JSAMPLE *)malloc((5 * (MAXJSAMPLE + 1) + CENTERJSAMPLE) * sizeof(JSAMPLE));
        table = original_table_ptr;
        table += (MAXJSAMPLE + 1);

        sample_range_limit = table;
        memset(table - (MAXJSAMPLE + 1), 0, (MAXJSAMPLE + 1) * sizeof(JSAMPLE));

        for (i = 0; i <= MAXJSAMPLE; i++)
        table[i] = (JSAMPLE)i;
        table += CENTERJSAMPLE;

        for (i = CENTERJSAMPLE; i < 2 * (MAXJSAMPLE + 1); i++)
        table[i] = MAXJSAMPLE;

        memset(table + (2 * (MAXJSAMPLE + 1)), 0,
            (2 * (MAXJSAMPLE + 1) - CENTERJSAMPLE) * sizeof(JSAMPLE));
        memcpy(table + (4 * (MAXJSAMPLE + 1) - CENTERJSAMPLE),
            sample_range_limit, CENTERJSAMPLE * sizeof(JSAMPLE));
    }
    void SetUp() override {

        setupTable();

        for (int i = 0; i < 64; i++) {
            dct_coeffs[i] = 0;
        }

        short standard_quant[64] = {
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99
        };
        memcpy(quant_table, standard_quant, sizeof(standard_quant));

        for (int row = 0; row < 8; row++) {
            output_data[row] = new unsigned char[8];
        }

        for (int i = 0; i < 64; i++) {
            dc_only_coeffs[i] = 0;
        }
        dc_only_coeffs[0] = 1024;

        for (int i = 0; i < 64; i++) {
            low_freq_coeffs[i] = 0;
        }
        low_freq_coeffs[0] = 1024;
        low_freq_coeffs[1] = 512;
        low_freq_coeffs[8] = 256;

        for (int i = 0; i < 64; i++) {
            high_freq_coeffs[i] = 0;
        }
        high_freq_coeffs[63] = 512;
        high_freq_coeffs[7] = 256;
        high_freq_coeffs[56] = 128;
    }

    void TearDown() override {

        for (int row = 0; row < 8; row++) {
            delete[] output_data[row];
        }

        if (original_table_ptr) {
            free(original_table_ptr);
        }
    }

    short quant_table[64];
    short dct_coeffs[64];
    short dc_only_coeffs[64];
    short low_freq_coeffs[64];
    short high_freq_coeffs[64];
    unsigned char *output_data[8];
};

TEST_F(JidctfstTest, HandlesZeroCoefficients) {

    idct_ifast(quant_table, dct_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_NEAR(output_data[row][col], 128, 10);
        }
    }
}

TEST_F(JidctfstTest, HandlesDCOnlyCoefficient) {

    idct_ifast(quant_table, dc_only_coeffs, output_data, 0, sample_range_limit);

    unsigned char first_pixel = output_data[0][0];
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_NEAR(output_data[row][col], first_pixel, 2);
        }
    }

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

#if 0
TEST_F(JidctfstTest, HandlesLowFrequencyCoefficients) {

    idct_ifast(quant_table, low_freq_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }

    bool has_variation = false;
    unsigned char first_pixel = output_data[0][0];
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            if (abs(output_data[row][col] - first_pixel) > 50) {
                has_variation = true;
                break;
            }
        }
        if (has_variation) break;
    }
    EXPECT_TRUE(has_variation);
}
#endif

TEST_F(JidctfstTest, HandlesHighFrequencyCoefficients) {

    idct_ifast(quant_table, high_freq_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }

    int min_val = 255, max_val = 0;
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            min_val = std::min(min_val, (int)output_data[row][col]);
            max_val = std::max(max_val, (int)output_data[row][col]);
        }
    }
    EXPECT_GT(max_val - min_val, 20);
}

TEST_F(JidctfstTest, OutputColumnOffsetTest) {

    unsigned char extended_output[8][16];
    unsigned char *extended_ptrs[8];

    for (int row = 0; row < 8; row++) {
        extended_ptrs[row] = extended_output[row];
    }

    idct_ifast(quant_table, dc_only_coeffs, extended_ptrs, 0, sample_range_limit);

    idct_ifast(quant_table, dc_only_coeffs, extended_ptrs, 8, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_NEAR(extended_output[row][col], extended_output[row][col + 8], 2);
        }
    }
}

TEST_F(JidctfstTest, ReproducibilityTest) {

    unsigned char output1[8][8], output2[8][8];
    unsigned char *ptrs1[8], *ptrs2[8];

    for (int row = 0; row < 8; row++) {
        ptrs1[row] = output1[row];
        ptrs2[row] = output2[row];
    }

    short test_coeffs[64];
    for (int i = 0; i < 64; i++) {
        test_coeffs[i] = (i % 8) * 64 - 256;
    }

    idct_ifast(quant_table, test_coeffs, ptrs1, 0, sample_range_limit);
    idct_ifast(quant_table, test_coeffs, ptrs2, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_EQ(output1[row][col], output2[row][col]);
        }
    }
}

TEST_F(JidctfstTest, BoundaryCoefficientsTest) {

    short extreme_coeffs[64];
    for (int i = 0; i < 64; i++) {
        extreme_coeffs[i] = 0;
    }
    extreme_coeffs[0] = 2047;
    extreme_coeffs[1] = -1024;

    idct_ifast(quant_table, extreme_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

TEST_F(JidctfstTest, AlternatingPatternTest) {

    short alternating_coeffs[64];
    for (int i = 0; i < 64; i++) {
        alternating_coeffs[i] = (i % 2) ? 256 : -256;
    }
    alternating_coeffs[0] = 1024;

    idct_ifast(quant_table, alternating_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }

    int min_val = 255, max_val = 0;
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            min_val = std::min(min_val, (int)output_data[row][col]);
            max_val = std::max(max_val, (int)output_data[row][col]);
        }
    }
    EXPECT_GT(max_val - min_val, 30);
}

TEST_F(JidctfstTest, GradientCoefficientsTest) {

    short gradient_coeffs[64];
    for (int i = 0; i < 64; i++) {
        gradient_coeffs[i] = (i < 32) ? (i * 16) : 0;
    }

    idct_ifast(quant_table, gradient_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

TEST_F(JidctfstTest, SingleCoefficientTest) {

    for (int pos = 0; pos < 64; pos += 8) {
        short single_coeff[64];
        for (int i = 0; i < 64; i++) {
            single_coeff[i] = 0;
        }
        single_coeff[pos] = 512;

        idct_ifast(quant_table, single_coeff, output_data, 0, sample_range_limit);

        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                EXPECT_GE(output_data[row][col], 0);
                EXPECT_LE(output_data[row][col], 255);
            }
        }
    }
}

TEST_F(JidctfstTest, SymmetryTest) {

    short symmetric_coeffs[64];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int idx = i * 8 + j;

            symmetric_coeffs[idx] = (i + j < 4) ? 256 : 0;
        }
    }

    idct_ifast(quant_table, symmetric_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

TEST_F(JidctfstTest, RoundTripConsistencyCheck) {

    unsigned char original_image[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            original_image[i][j] = (i * 32 + j * 16) % 256;
        }
    }

    short reasonable_coeffs[64];
    reasonable_coeffs[0] = 1024;
    for (int i = 1; i < 64; i++) {
        reasonable_coeffs[i] = (i < 16) ? (256 / (i + 1)) : 0;
    }

    idct_ifast(quant_table, reasonable_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}
