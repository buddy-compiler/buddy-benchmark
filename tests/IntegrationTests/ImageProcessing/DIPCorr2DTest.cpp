#include "ImageProcessing/Kernels.h"
#include "Utils/Container.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <gtest/gtest.h>
#include <iostream>
#include <string>
// #include "../../UniTests/Test.h"

using namespace cv;
using namespace std;

// Declare the Corr2D C interface.
extern "C" {
void _mlir_ciface_corr_2d(MemRef<float, 2> *input, MemRef<float, 2> *kernel,
                          MemRef<float, 2> *output, unsigned int centerX,
                          unsigned int centerY, int boundaryOption);
}

// Fixture for testing the dip.corr_2d operation.
class FilterTest : public ::testing::Test {
protected:
  void SetUp()
  {
    inputImage = imread(imageName, IMREAD_GRAYSCALE);
  }

public:
  static void setImageName(std::string imageNameParam)
  {
    imageName = imageNameParam;
  }

  const Mat& getInputImage()
  {
    return inputImage;
  }

private:
  Mat inputImage;
  static std::string imageName;
};

bool equalImages(const Mat &img1, const Mat &img2) {
  if (img1.rows != img2.rows || img1.cols != img2.cols) {
    std::cout << "Produced outputs by DIP and OpenCV differ. Image dimensions "
                 "are not equal\n";
    return 0;
  }

  for (unsigned int y = 0; y < img1.rows; ++y) {
    for (unsigned int x = 0; x < img1.cols; ++x) {
      if (abs(img1.at<float>(x, y) - img2.at<float>(x, y)) > 10e-3) {
        std::cout << "Produced outputs by DIP and OpenCV differ.\n";
        return 0;
      }
    }
  }
  return 1;
}

void testKernelImpl(const Mat &inputImage, unsigned int kernelRows,
                    unsigned int kernelCols, float *kernelArray, unsigned int x,
                    unsigned int y) {
  // Define container sizes.
  intptr_t sizesInput[2] = {inputImage.rows, inputImage.cols};
  intptr_t sizesKernel[2] = {kernelRows, kernelCols};
  intptr_t sizesOutput[2] = {inputImage.rows, inputImage.cols};

  // Define input, kernel, and output.
  MemRef<float, 2> input(inputImage, sizesInput);
  MemRef<float, 2> kernel(kernelArray, sizesKernel);
  MemRef<float, 2> output(sizesOutput);

  for (int i = 0; i < inputImage.rows; i++)
    for (int j = 0; j < inputImage.cols; j++)
      output[i * inputImage.rows + j] = 0;

  Mat kernel1 = Mat(kernelRows, kernelCols, CV_32FC1, kernelArray);
  Mat opencvOutput;

  _mlir_ciface_corr_2d(&input, &kernel, &output, x, y, 0);

  filter2D(inputImage, opencvOutput, CV_32FC1, kernel1, cv::Point(x, y), 0.0,
           cv::BORDER_REPLICATE);

  // Define a cv::Mat with the output of corr_2d.
  Mat dipOutput(inputImage.rows, inputImage.cols, CV_32FC1, output.getData());

  if (!equalImages(dipOutput, opencvOutput)) {
    std::cout << "Different images produced by OpenCV and DIP for kernel :\n"
              << kernel1
              << "\n"
                 "when anchor point was : ("
              << x << ", " << y << ").\n";
    return;
  }
}

void testKernel(const Mat &inputImage, unsigned int kernelRows,
                unsigned int kernelCols, float *kernelArray) {
  for (unsigned int y = 0; y < kernelRows; ++y)
    for (unsigned int x = 0; x < kernelCols; ++x)
      testKernelImpl(inputImage, kernelRows, kernelCols, kernelArray, x, y);
}

TEST_F(FilterTest, OpenCVComparison) {
  for (auto kernel : kernelMap)
    testKernel(getInputImage(), get<1>(kernel.second), get<2>(kernel.second),
               get<0>(kernel.second));

  // testKernel(getInputImage(), 3, 3, laplacianKernelAlign);
  // ASSERT_EQ(2, 2);
}

string FilterTest::imageName = "";

int main(int argc, char **argv)
{
  FilterTest::setImageName(argv[1]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
