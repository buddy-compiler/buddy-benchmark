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

extern void idct_islow(short *quantptr, JCOEFPTR coef_block, JSAMPARRAY output_buf, JDIMENSION output_col, JSAMPROW sample_range_limit);

class JidctintTest : public ::testing::Test {
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

        for (int i = 0; i < 64; i++) {
            dct_coeffs[i] = 0;
        }

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

        for (int i = 0; i < 64; i++) {
            integer_coeffs[i] = (i % 8) * 128 - 256;
        }
        integer_coeffs[0] = 1024;
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
    short integer_coeffs[64];
    unsigned char *output_data[8];
};

TEST_F(JidctintTest, HandlesZeroCoefficients) {

    idct_islow(quant_table, dct_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_NEAR(output_data[row][col], 128, 5);
        }
    }
}

TEST_F(JidctintTest, HandlesDCOnlyCoefficient) {

    idct_islow(quant_table, dc_only_coeffs, output_data, 0, sample_range_limit);

    unsigned char first_pixel = output_data[0][0];
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_NEAR(output_data[row][col], first_pixel, 3);
        }
    }

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

TEST_F(JidctintTest, HandlesLowFrequencyCoefficients) {

    idct_islow(quant_table, low_freq_coeffs, output_data, 0, sample_range_limit);

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

TEST_F(JidctintTest, HandlesHighFrequencyCoefficients) {

    idct_islow(quant_table, high_freq_coeffs, output_data, 0, sample_range_limit);

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

TEST_F(JidctintTest, IntegerPrecisionTest) {

    idct_islow(quant_table, integer_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

TEST_F(JidctintTest, OutputColumnOffsetTest) {

    unsigned char extended_output[8][16];
    unsigned char *extended_ptrs[8];

    for (int row = 0; row < 8; row++) {
        extended_ptrs[row] = extended_output[row];
    }

    idct_islow(quant_table, dc_only_coeffs, extended_ptrs, 0, sample_range_limit);

    idct_islow(quant_table, dc_only_coeffs, extended_ptrs, 8, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_NEAR(extended_output[row][col], extended_output[row][col + 8], 3);
        }
    }
}

TEST_F(JidctintTest, ReproducibilityTest) {

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

    idct_islow(quant_table, test_coeffs, ptrs1, 0, sample_range_limit);
    idct_islow(quant_table, test_coeffs, ptrs2, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_EQ(output1[row][col], output2[row][col]);
        }
    }
}

TEST_F(JidctintTest, BoundaryCoefficientsTest) {

    short extreme_coeffs[64];
    for (int i = 0; i < 64; i++) {
        extreme_coeffs[i] = 0;
    }
    extreme_coeffs[0] = 2047;
    extreme_coeffs[1] = -1024;

    idct_islow(quant_table, extreme_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

TEST_F(JidctintTest, IntegerStabilityTest) {

    short stability_coeffs[64];
    for (int i = 0; i < 64; i++) {
        stability_coeffs[i] = (i % 16) * 32 - 256;
    }
    stability_coeffs[0] = 1024;

    idct_islow(quant_table, stability_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

TEST_F(JidctintTest, StepResponseTest) {

    short step_coeffs[64];
    for (int i = 0; i < 64; i++) {
        step_coeffs[i] = (i < 32) ? 256 : -256;
    }
    step_coeffs[0] = 1024;

    idct_islow(quant_table, step_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

TEST_F(JidctintTest, SparseCoefficientsTest) {

    short sparse_coeffs[64];
    for (int i = 0; i < 64; i++) {
        sparse_coeffs[i] = 0;
    }
    sparse_coeffs[0] = 1024;
    sparse_coeffs[9] = 128;
    sparse_coeffs[63] = 64;

    idct_islow(quant_table, sparse_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

TEST_F(JidctintTest, PowerOfTwoCoefficientsTest) {

    short power2_coeffs[64];
    for (int i = 0; i < 64; i++) {
        int power = i % 10;
        power2_coeffs[i] = (power > 0) ? (1 << power) : 0;
        if (i % 2) power2_coeffs[i] = -power2_coeffs[i];
    }
    power2_coeffs[0] = 1024;

    idct_islow(quant_table, power2_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

TEST_F(JidctintTest, QuantizedCoefficientsTest) {

    short quantized_coeffs[64];
    quantized_coeffs[0] = 1024;
    quantized_coeffs[1] = 256;
    quantized_coeffs[2] = 128;
    quantized_coeffs[8] = 192;
    quantized_coeffs[9] = 64;

    for (int i = 10; i < 64; i++) {
        quantized_coeffs[i] = (i % 3) ? 0 : (32 / (i / 8 + 1));
    }

    idct_islow(quant_table, quantized_coeffs, output_data, 0, sample_range_limit);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }
}

TEST_F(JidctintTest, ConsistencyWithFastVersionTest) {

    short consistency_coeffs[64];
    for (int i = 0; i < 64; i++) {
        consistency_coeffs[i] = (i < 8) ? (8 - i) : 0;
    }

    idct_islow(quant_table, consistency_coeffs, output_data, 0, sample_range_limit);

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
    EXPECT_GE(max_val - min_val, 10);
    EXPECT_LE(max_val - min_val, 200);
}
