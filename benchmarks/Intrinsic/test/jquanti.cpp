#include <gtest/gtest.h>
#include <vector>
#include <cstring>

typedef short DCTELEM;
typedef short JCOEF;
typedef JCOEF *JCOEFPTR;

typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef unsigned int JDIMENSION;

#define DCTSIZE         8
#define DCTSIZE2        64
#define MAXJSAMPLE      255
#define CENTERJSAMPLE   128

extern void quantize(JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);
extern void convsamp(JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

enum class TestPattern {
    ZEROS,
    ONES,
    MAX_VALUES,
    RANDOM,
    DC_ONLY,
    AC_ONLY
};

class QuantizeTest : public ::testing::Test {
protected:
    static constexpr int DCT_SIZE = 8;
    static constexpr int DCT_SIZE2 = DCT_SIZE * DCT_SIZE;

    void SetUp() override {

        workspace_.resize(DCT_SIZE2);
        divisors_.resize(DCT_SIZE2 * 4);
        coef_block_ref_.resize(DCT_SIZE2);
        coef_block_test_.resize(DCT_SIZE2);
    }

    void TearDown() override {

    }

    void computeReciprocal(uint16_t divisor, DCTELEM *dtbl, int index) {
        if (divisor == 1) {
            dtbl[index + DCT_SIZE2 * 0] = 1;
            dtbl[index + DCT_SIZE2 * 1] = 0;
            dtbl[index + DCT_SIZE2 * 2] = 1;
            dtbl[index + DCT_SIZE2 * 3] = -(int)(sizeof(DCTELEM) * 8);
            return;
        }

        int b = 0;
        uint16_t val = divisor;
        while (val >>= 1) b++;

        int r = sizeof(DCTELEM) * 8 + b;
        uint64_t fq = ((uint64_t)1 << r) / divisor;
        uint64_t fr = ((uint64_t)1 << r) % divisor;

        uint32_t c = divisor / 2;

        if (fr == 0) {
            fq >>= 1;
            r--;
        } else if (fr <= (divisor / 2U)) {
            c++;
        } else {
            fq++;
        }

        dtbl[index + DCT_SIZE2 * 0] = (DCTELEM)fq;
        dtbl[index + DCT_SIZE2 * 1] = (DCTELEM)c;
        dtbl[index + DCT_SIZE2 * 2] = (DCTELEM)(1 << (sizeof(DCTELEM) * 8 * 2 - r));
        dtbl[index + DCT_SIZE2 * 3] = (DCTELEM)r - sizeof(DCTELEM) * 8;
    }

    void generateStandardQuantTable(std::vector<uint16_t>& quant_table, int quality = 50) {

        static const uint16_t std_luminance_quant_tbl[DCT_SIZE2] = {
            16,  11,  10,  16,  24,  40,  51,  61,
            12,  12,  14,  19,  26,  58,  60,  55,
            14,  13,  16,  24,  40,  57,  69,  56,
            14,  17,  22,  29,  51,  87,  80,  62,
            18,  22,  37,  56,  68, 109, 103,  77,
            24,  35,  55,  64,  81, 104, 113,  92,
            49,  64,  78,  87, 103, 121, 120, 101,
            72,  92,  95,  98, 112, 100, 103,  99
        };

        quant_table.resize(DCT_SIZE2);

        int scale_factor;
        if (quality < 50) {
            scale_factor = 5000 / quality;
        } else {
            scale_factor = 200 - quality * 2;
        }

        for (int i = 0; i < DCT_SIZE2; i++) {
            int temp = (std_luminance_quant_tbl[i] * scale_factor + 50) / 100;
            quant_table[i] = std::max(1, std::min(255, temp));
        }
    }

    void generateTestCoefficients(std::vector<DCTELEM>& coeffs, TestPattern pattern) {
        coeffs.resize(DCT_SIZE2);

        switch (pattern) {
        case TestPattern::ZEROS:
            std::fill(coeffs.begin(), coeffs.end(), 0);
            break;

        case TestPattern::ONES:
            std::fill(coeffs.begin(), coeffs.end(), 1);
            break;

        case TestPattern::MAX_VALUES:
            for (int i = 0; i < DCT_SIZE2; i++) {
                coeffs[i] = (i % 2 == 0) ? 2047 : -2047;
            }
            break;

        case TestPattern::RANDOM:
            {
                for (int i = 0; i < DCT_SIZE2; i++) {
                    int val = ((i * 113 + 37) % 4096) - 2048;
                    coeffs[i] = val;
                }
            }
            break;

        case TestPattern::DC_ONLY:
            std::fill(coeffs.begin(), coeffs.end(), 0);
            coeffs[0] = 1024;
            break;

        case TestPattern::AC_ONLY:
            {
                coeffs[0] = 0;
                for (int i = 1; i < DCT_SIZE2; i++) {
                    int val = ((i * 73 + 19) % 1025) - 512;
                    coeffs[i] = val;
                }
            }
            break;
        }
    }

    bool validateQuantizeResult(const std::vector<JCOEF>& result,
                              const std::vector<DCTELEM>& input,
                              const std::vector<uint16_t>& quant_table) {
        for (int i = 0; i < DCT_SIZE2; i++) {

            int expected = input[i];
            int divisor = quant_table[i];

            if (expected < 0) {
                expected = -expected;
                expected = (expected + divisor/2) / divisor;
                expected = -expected;
            } else {
                expected = (expected + divisor/2) / divisor;
            }

            int diff = std::abs(result[i] - expected);
            if (diff > 1) {
                return false;
            }
        }
        return true;
    }

    std::vector<DCTELEM> workspace_;
    std::vector<DCTELEM> divisors_;
    std::vector<JCOEF> coef_block_ref_;
    std::vector<JCOEF> coef_block_test_;
};

TEST_F(QuantizeTest, BasicFunctionality) {

    std::vector<uint16_t> quant_table;
    generateStandardQuantTable(quant_table, 50);

    for (int i = 0; i < DCT_SIZE2; i++) {
        computeReciprocal(quant_table[i], divisors_.data(), i);
    }

    generateTestCoefficients(workspace_, TestPattern::RANDOM);

    quantize(coef_block_ref_.data(), divisors_.data(), workspace_.data());

    EXPECT_TRUE(validateQuantizeResult(coef_block_ref_, workspace_, quant_table));
}

TEST_F(QuantizeTest, BoundaryConditions) {
    std::vector<uint16_t> quant_table;
    generateStandardQuantTable(quant_table, 50);

    for (int i = 0; i < DCT_SIZE2; i++) {
        computeReciprocal(quant_table[i], divisors_.data(), i);
    }

    generateTestCoefficients(workspace_, TestPattern::ZEROS);
    quantize(coef_block_ref_.data(), divisors_.data(), workspace_.data());

    for (int i = 0; i < DCT_SIZE2; i++) {
        EXPECT_EQ(coef_block_ref_[i], 0) << "Zero input should produce zero output at index " << i;
    }

    generateTestCoefficients(workspace_, TestPattern::MAX_VALUES);
    quantize(coef_block_ref_.data(), divisors_.data(), workspace_.data());

    for (int i = 0; i < DCT_SIZE2; i++) {
        EXPECT_GE(coef_block_ref_[i], -32768) << "Output underflow at index " << i;
        EXPECT_LE(coef_block_ref_[i], 32767) << "Output overflow at index " << i;
    }
}

TEST_F(QuantizeTest, DifferentQualityFactors) {
    generateTestCoefficients(workspace_, TestPattern::RANDOM);

    std::vector<int> qualities = {10, 25, 50, 75};

    for (int quality : qualities) {
        std::vector<uint16_t> quant_table;
        generateStandardQuantTable(quant_table, quality);

        for (int i = 0; i < DCT_SIZE2; i++) {
            computeReciprocal(quant_table[i], divisors_.data(), i);
        }

        quantize(coef_block_ref_.data(), divisors_.data(), workspace_.data());

        EXPECT_TRUE(validateQuantizeResult(coef_block_ref_, workspace_, quant_table))
            << "Validation failed for quality factor " << quality;
    }
}

TEST_F(QuantizeTest, DCAndACComponents) {
    std::vector<uint16_t> quant_table;
    generateStandardQuantTable(quant_table, 50);

    for (int i = 0; i < DCT_SIZE2; i++) {
        computeReciprocal(quant_table[i], divisors_.data(), i);
    }

    generateTestCoefficients(workspace_, TestPattern::DC_ONLY);
    quantize(coef_block_ref_.data(), divisors_.data(), workspace_.data());

    EXPECT_NE(coef_block_ref_[0], 0) << "DC component should not be zero";
    for (int i = 1; i < DCT_SIZE2; i++) {
        EXPECT_EQ(coef_block_ref_[i], 0) << "AC component should be zero at index " << i;
    }

    generateTestCoefficients(workspace_, TestPattern::AC_ONLY);
    quantize(coef_block_ref_.data(), divisors_.data(), workspace_.data());

    EXPECT_EQ(coef_block_ref_[0], 0) << "DC component should be zero";
}

enum SamplePattern {
    SAMPLE_ZEROS,
    SAMPLE_MAX_VALUES,
    SAMPLE_MID_VALUES,
    SAMPLE_GRADIENT,
    SAMPLE_CHECKERBOARD,
    SAMPLE_RANDOM
};

class ConvSampTest : public ::testing::Test {
protected:
    static constexpr int DCT_SIZE = 8;
    static constexpr int DCT_SIZE2 = DCT_SIZE * DCT_SIZE;
    static constexpr int CENTER_SAMPLE = 128;

    void SetUp() override {

        sample_data_.resize(DCT_SIZE);
        sample_buffer_.resize(DCT_SIZE * (DCT_SIZE + 4));
        workspace_.resize(DCT_SIZE2);
        expected_output_.resize(DCT_SIZE2);

        for (int i = 0; i < DCT_SIZE; i++) {
            sample_data_[i] = &sample_buffer_[i * (DCT_SIZE + 4)];
        }
    }

    void TearDown() override {

    }

    void generateSampleData(SamplePattern pattern, JDIMENSION start_col = 0) {
        int buffer_width = DCT_SIZE + 4;

        switch (pattern) {
        case SAMPLE_ZEROS:
            std::fill(sample_buffer_.begin(), sample_buffer_.end(), 0);
            break;

        case SAMPLE_MAX_VALUES:
            std::fill(sample_buffer_.begin(), sample_buffer_.end(), MAXJSAMPLE);
            break;

        case SAMPLE_MID_VALUES:
            std::fill(sample_buffer_.begin(), sample_buffer_.end(), CENTERJSAMPLE);
            break;

        case SAMPLE_GRADIENT:
            for (int row = 0; row < DCT_SIZE; row++) {
                for (int col = 0; col < buffer_width; col++) {
                    sample_buffer_[row * buffer_width + col] =
                        (JSAMPLE)((row * DCT_SIZE + col) * MAXJSAMPLE / (DCT_SIZE2 - 1));
                }
            }
            break;

        case SAMPLE_CHECKERBOARD:
            for (int row = 0; row < DCT_SIZE; row++) {
                for (int col = 0; col < buffer_width; col++) {
                    sample_buffer_[row * buffer_width + col] =
                        ((row + col) % 2 == 0) ? 0 : MAXJSAMPLE;
                }
            }
            break;

        case SAMPLE_RANDOM:
            for (int row = 0; row < DCT_SIZE; row++) {
                for (int col = 0; col < buffer_width; col++) {

                    sample_buffer_[row * buffer_width + col] =
                        (JSAMPLE)((row * 73 + col * 97 + 23) % (MAXJSAMPLE + 1));
                }
            }
            break;
        }
    }

    void computeExpectedOutput(JDIMENSION start_col) {
        for (int row = 0; row < DCT_SIZE; row++) {
            for (int col = 0; col < DCT_SIZE; col++) {
                JSAMPLE input_val = sample_data_[row][start_col + col];
                expected_output_[row * DCT_SIZE + col] = (DCTELEM)(input_val - CENTERJSAMPLE);
            }
        }
    }

    bool validateOutput() {
        for (int i = 0; i < DCT_SIZE2; i++) {
            if (workspace_[i] != expected_output_[i]) {
                return false;
            }
        }
        return true;
    }

    bool validateOutputRange() {
        for (int i = 0; i < DCT_SIZE2; i++) {
            if (workspace_[i] < -CENTERJSAMPLE || workspace_[i] > (MAXJSAMPLE - CENTERJSAMPLE)) {
                return false;
            }
        }
        return true;
    }

    std::vector<JSAMPROW> sample_data_;
    std::vector<JSAMPLE> sample_buffer_;
    std::vector<DCTELEM> workspace_;
    std::vector<DCTELEM> expected_output_;
};

TEST_F(ConvSampTest, BasicFunctionality) {
    generateSampleData(SAMPLE_GRADIENT);
    computeExpectedOutput(0);

    convsamp(sample_data_.data(), 0, workspace_.data());

    EXPECT_TRUE(validateOutput()) << "Output does not match expected values";
    EXPECT_TRUE(validateOutputRange()) << "Output values are out of valid range";
}

TEST_F(ConvSampTest, BoundaryValues) {

    generateSampleData(SAMPLE_ZEROS);
    computeExpectedOutput(0);

    convsamp(sample_data_.data(), 0, workspace_.data());

    EXPECT_TRUE(validateOutput()) << "Zero input test failed";
    for (int i = 0; i < DCT_SIZE2; i++) {
        EXPECT_EQ(workspace_[i], -CENTER_SAMPLE) << "Zero input should produce -128 at index " << i;
    }

    generateSampleData(SAMPLE_MAX_VALUES);
    computeExpectedOutput(0);

    convsamp(sample_data_.data(), 0, workspace_.data());

    EXPECT_TRUE(validateOutput()) << "Max value input test failed";
    for (int i = 0; i < DCT_SIZE2; i++) {
        EXPECT_EQ(workspace_[i], MAXJSAMPLE - CENTER_SAMPLE) << "Max input should produce 127 at index " << i;
    }

    generateSampleData(SAMPLE_MID_VALUES);
    computeExpectedOutput(0);

    convsamp(sample_data_.data(), 0, workspace_.data());

    EXPECT_TRUE(validateOutput()) << "Mid value input test failed";
    for (int i = 0; i < DCT_SIZE2; i++) {
        EXPECT_EQ(workspace_[i], 0) << "Mid input should produce 0 at index " << i;
    }
}

TEST_F(ConvSampTest, DifferentStartColumns) {
    generateSampleData(SAMPLE_GRADIENT);

    std::vector<JDIMENSION> start_cols = {0, 1, 2, 3, 4};

    for (JDIMENSION start_col : start_cols) {
        computeExpectedOutput(start_col);
        convsamp(sample_data_.data(), start_col, workspace_.data());

        EXPECT_TRUE(validateOutput()) << "Validation failed for start_col " << start_col;
        EXPECT_TRUE(validateOutputRange()) << "Range validation failed for start_col " << start_col;
    }
}

TEST_F(ConvSampTest, DifferentDataPatterns) {
    std::vector<SamplePattern> patterns = {
        SAMPLE_ZEROS,
        SAMPLE_MAX_VALUES,
        SAMPLE_MID_VALUES,
        SAMPLE_GRADIENT,
        SAMPLE_CHECKERBOARD,
        SAMPLE_RANDOM
    };

    for (SamplePattern pattern : patterns) {
        generateSampleData(pattern);
        computeExpectedOutput(0);

        convsamp(sample_data_.data(), 0, workspace_.data());

        EXPECT_TRUE(validateOutput()) << "Pattern test failed for pattern " << (int)pattern;
        EXPECT_TRUE(validateOutputRange()) << "Range test failed for pattern " << (int)pattern;
    }
}

TEST_F(ConvSampTest, CheckerboardPattern) {
    generateSampleData(SAMPLE_CHECKERBOARD);
    computeExpectedOutput(0);

    convsamp(sample_data_.data(), 0, workspace_.data());

    EXPECT_TRUE(validateOutput()) << "Checkerboard pattern test failed";

    for (int row = 0; row < DCT_SIZE; row++) {
        for (int col = 0; col < DCT_SIZE; col++) {
            int index = row * DCT_SIZE + col;
            DCTELEM expected = ((row + col) % 2 == 0) ? -CENTERJSAMPLE : (MAXJSAMPLE - CENTERJSAMPLE);
            EXPECT_EQ(workspace_[index], expected)
                << "Checkerboard value incorrect at (" << row << "," << col << ")";
        }
    }
}

TEST_F(ConvSampTest, DataIntegrity) {
    generateSampleData(SAMPLE_RANDOM);

    const JSAMPLE GUARD_VALUE = 0xAA;
    for (int row = 0; row < DCT_SIZE; row++) {
        sample_data_[row][DCT_SIZE] = GUARD_VALUE;
    }

    computeExpectedOutput(0);
    convsamp(sample_data_.data(), 0, workspace_.data());

    EXPECT_TRUE(validateOutput()) << "Data integrity test failed";

    for (int row = 0; row < DCT_SIZE; row++) {
        EXPECT_EQ(sample_data_[row][DCT_SIZE], GUARD_VALUE)
            << "Guard value was modified at row " << row;
    }
}

TEST_F(ConvSampTest, WorkspaceSize) {
    generateSampleData(SAMPLE_GRADIENT);
    computeExpectedOutput(0);

    std::fill(workspace_.begin(), workspace_.end(), 0x5555);

    convsamp(sample_data_.data(), 0, workspace_.data());

    for (int i = 0; i < DCT_SIZE2; i++) {
        EXPECT_NE(workspace_[i], (DCTELEM)0x5555)
            << "Workspace position " << i << " was not written";
        EXPECT_EQ(workspace_[i], expected_output_[i])
            << "Workspace position " << i << " has incorrect value";
    }
}
