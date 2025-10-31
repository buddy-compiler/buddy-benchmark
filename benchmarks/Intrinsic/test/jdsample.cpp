#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cstring>

typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef unsigned int JDIMENSION;

extern void h2v1_upsample(int max_v_samp_factor,
                   JDIMENSION output_width,
                   JSAMPARRAY input_data,
                   JSAMPARRAY *output_data_ptr);

extern void h2v2_upsample(int max_v_samp_factor,
                   JDIMENSION output_width,
                   JSAMPARRAY input_data,
                   JSAMPARRAY *output_data_ptr);

extern void h2v1_fancy_upsample(int max_v_samp_factor,
                         JDIMENSION downsampled_width,
                         JSAMPARRAY input_data,
                         JSAMPARRAY *output_data_ptr);

extern void h2v2_fancy_upsample(int max_v_samp_factor,
                         JDIMENSION downsampled_width,
                         JSAMPARRAY input_data,
                         JSAMPARRAY *output_data_ptr);

class UpsampleTest : public ::testing::Test {
protected:
    void SetUp() override {

        max_v_samp_factor = 2;
        downsampled_width = 16;
        output_width = 32;

        AllocateInputData();
        AllocateOutputData();
    }

    void TearDown() override {

        FreeInputData();
        FreeOutputData();
    }

    void AllocateInputData() {

        input_data = new JSAMPROW[max_v_samp_factor];
        for (int i = 0; i < max_v_samp_factor; i++) {
            input_data[i] = new JSAMPLE[downsampled_width];
        }
    }

    void AllocateOutputData() {

        output_data = new JSAMPROW[max_v_samp_factor * 2];
        for (int i = 0; i < max_v_samp_factor * 2; i++) {
            output_data[i] = new JSAMPLE[output_width];

            memset(output_data[i], 0, output_width * sizeof(JSAMPLE));
        }
    }

    void FreeInputData() {
        if (input_data) {
            for (int i = 0; i < max_v_samp_factor; i++) {
                delete[] input_data[i];
            }
            delete[] input_data;
            input_data = nullptr;
        }
    }

    void FreeOutputData() {
        if (output_data) {
            for (int i = 0; i < max_v_samp_factor * 2; i++) {
                delete[] output_data[i];
            }
            delete[] output_data;
            output_data = nullptr;
        }
    }

    void InitializeInputData(const std::vector<std::vector<JSAMPLE>>& data) {
        for (size_t row = 0; row < data.size() && row < max_v_samp_factor; row++) {
            for (size_t col = 0; col < data[row].size() && col < downsampled_width; col++) {
                input_data[row][col] = data[row][col];
            }
        }
    }

    int max_v_samp_factor;
    JDIMENSION downsampled_width;
    JDIMENSION output_width;
    JSAMPARRAY input_data;
    JSAMPARRAY output_data;
};

TEST_F(UpsampleTest, H2V1_UpsampleBasicFunctionality) {

    std::vector<std::vector<JSAMPLE>> test_input = {
        {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160},
        {15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165}
    };
    InitializeInputData(test_input);

    JSAMPARRAY output_ptr = output_data;
    h2v1_upsample(max_v_samp_factor, output_width, input_data, &output_ptr);

    EXPECT_EQ(output_data[0][0], 10);
    EXPECT_EQ(output_data[0][1], 10);
    EXPECT_EQ(output_data[0][2], 20);
    EXPECT_EQ(output_data[0][3], 20);
    EXPECT_EQ(output_data[0][4], 30);
    EXPECT_EQ(output_data[0][5], 30);

    EXPECT_EQ(output_data[0][30], 160);
    EXPECT_EQ(output_data[0][31], 160);

    EXPECT_EQ(output_data[1][0], 15);
    EXPECT_EQ(output_data[1][1], 15);
    EXPECT_EQ(output_data[1][2], 25);
    EXPECT_EQ(output_data[1][3], 25);

    EXPECT_EQ(output_data[1][30], 165);
    EXPECT_EQ(output_data[1][31], 165);
}

TEST_F(UpsampleTest, H2V2_UpsampleBasicFunctionality) {

    std::vector<std::vector<JSAMPLE>> test_input = {
        {50, 100, 150, 200},
        {60, 110, 160, 210}
    };
    InitializeInputData(test_input);

    JSAMPARRAY output_ptr = output_data;
    h2v2_upsample(max_v_samp_factor, output_width, input_data, &output_ptr);

    for (int col = 0; col < 8; col++) {
        EXPECT_EQ(output_data[0][col], output_data[1][col]);
    }

    for (int col = 0; col < 8; col++) {
        EXPECT_EQ(output_data[2][col], output_data[3][col]);
    }

    EXPECT_EQ(output_data[0][0], 50);
    EXPECT_EQ(output_data[0][1], 50);
    EXPECT_EQ(output_data[0][2], 100);
    EXPECT_EQ(output_data[0][3], 100);
}

TEST_F(UpsampleTest, H2V1_FancyUpsampleBasicFunctionality) {

    std::vector<std::vector<JSAMPLE>> test_input = {
        {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160},
        {15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165}
    };
    InitializeInputData(test_input);

    JSAMPARRAY output_ptr = output_data;
    h2v1_fancy_upsample(max_v_samp_factor, downsampled_width, input_data, &output_ptr);

    EXPECT_EQ(output_data[0][0], 10);

    EXPECT_EQ(output_data[0][1], 13);

    EXPECT_EQ(output_data[0][30], 157);
    EXPECT_EQ(output_data[0][31], 160);

    EXPECT_EQ(output_data[1][0], 15);
    EXPECT_EQ(output_data[1][31], 165);
}

TEST_F(UpsampleTest, H2V2_FancyUpsampleBasicFunctionality) {

    JSAMPARRAY extended_input = new JSAMPROW[max_v_samp_factor + 2];
    for (int i = 0; i < max_v_samp_factor + 2; i++) {
        extended_input[i] = new JSAMPLE[downsampled_width];
    }

    std::vector<std::vector<JSAMPLE>> test_input = {
        {30, 70, 110, 150},
        {40, 80, 120, 160},
        {50, 90, 130, 170},
        {60, 100, 140, 180}
    };

    for (size_t row = 0; row < test_input.size(); row++) {
        for (size_t col = 0; col < test_input[row].size(); col++) {
            extended_input[row][col] = test_input[row][col];
        }
    }

    JSAMPARRAY input_ptr = extended_input + 1;
    JSAMPARRAY output_ptr = output_data;

    h2v2_fancy_upsample(max_v_samp_factor, downsampled_width, input_ptr, &output_ptr);

    EXPECT_GT(output_data[0][0], 0);
    EXPECT_LT(output_data[0][0], 255);

    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 8; col++) {
            EXPECT_GE(output_data[row][col], 0);
            EXPECT_LE(output_data[row][col], 255);
        }
    }

    for (int i = 0; i < max_v_samp_factor + 2; i++) {
        delete[] extended_input[i];
    }
    delete[] extended_input;
}

TEST_F(UpsampleTest, ExtremeValues) {
    std::vector<std::vector<JSAMPLE>> test_input = {
        {0, 255, 0, 255},
        {255, 0, 255, 0}
    };
    InitializeInputData(test_input);

    JSAMPARRAY output_ptr = output_data;
    h2v1_upsample(max_v_samp_factor, output_width, input_data, &output_ptr);

    EXPECT_EQ(output_data[0][0], 0);
    EXPECT_EQ(output_data[0][1], 0);
    EXPECT_EQ(output_data[0][2], 255);
    EXPECT_EQ(output_data[0][3], 255);

    EXPECT_EQ(output_data[1][0], 255);
    EXPECT_EQ(output_data[1][1], 255);
    EXPECT_EQ(output_data[1][2], 0);
    EXPECT_EQ(output_data[1][3], 0);
}

TEST_F(UpsampleTest, ParameterValidation) {
    std::vector<std::vector<JSAMPLE>> test_input = {
        {100, 150, 200, 250},
        {110, 160, 210, 255}
    };
    InitializeInputData(test_input);

    JSAMPARRAY output_ptr = output_data;
    h2v1_upsample(1, output_width, input_data, &output_ptr);

    EXPECT_EQ(output_data[0][0], 100);
    EXPECT_EQ(output_data[0][1], 100);
    EXPECT_EQ(output_data[0][2], 150);
    EXPECT_EQ(output_data[0][3], 150);
}

TEST_F(UpsampleTest, PerformanceBaseline) {

    const int large_width = 64;
    const int large_output_width = 128;

    JSAMPARRAY large_input = new JSAMPROW[max_v_samp_factor];
    JSAMPARRAY large_output = new JSAMPROW[max_v_samp_factor * 2];

    for (int i = 0; i < max_v_samp_factor; i++) {
        large_input[i] = new JSAMPLE[large_width];

        for (int j = 0; j < large_width; j++) {
            large_input[i][j] = (JSAMPLE)((i * large_width + j) % 256);
        }
    }

    for (int i = 0; i < max_v_samp_factor * 2; i++) {
        large_output[i] = new JSAMPLE[large_output_width];
        memset(large_output[i], 0, large_output_width * sizeof(JSAMPLE));
    }

    JSAMPARRAY output_ptr = large_output;
    for (int iter = 0; iter < 1000; iter++) {
        h2v1_upsample(max_v_samp_factor, large_output_width, large_input, &output_ptr);
    }

    EXPECT_EQ(large_output[0][0], large_input[0][0]);
    EXPECT_EQ(large_output[0][1], large_input[0][0]);

    for (int i = 0; i < max_v_samp_factor; i++) {
        delete[] large_input[i];
    }
    delete[] large_input;

    for (int i = 0; i < max_v_samp_factor * 2; i++) {
        delete[] large_output[i];
    }
    delete[] large_output;
}
