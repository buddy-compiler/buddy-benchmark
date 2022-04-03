#include "ImageProcessing/Kernels.h"
#include "Utils/Container.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <gtest/gtest.h>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// Declare the Corr2D C interface.
extern "C" {
void _mlir_ciface_corr_2d_constant_padding(MemRef<float, 2> *input, MemRef<float, 2> *kernel,
                                           MemRef<float, 2> *output, unsigned int centerX,
                                           unsigned int centerY, float constantValue);
}

// Fixture for testing the dip.corr_2d operation.
class FilterTest : public ::testing::Test {
public:
  static void setImageNames(int argc, char **argv)
  {
    if (argc > 1)
      testImageName = argv[1];
    else 
      {
        testImageName = "../../benchmarks/ImageProcessing/Images/YuTu.png";
        std::cout << "Reached here\n";
      }
  }

  const std::string getTestImageName()
  {
    return testImageName;
  }

private:
  static std::string testImageName;
};

std::string FilterTest::testImageName;

bool equalImages(const Mat &img1, const Mat &img2) {
  if (img1.rows != img2.rows || img1.cols != img2.cols) {
    std::cout << "Produced outputs by DIP and OpenCV differ. Image dimensions "
                 "are not equal\n";
    return 0;
  }

  for (unsigned int y = 0; y < img1.rows; ++y) {
    for (unsigned int x = 0; x < img1.cols; ++x) {
      if (abs(img1.at<float>(x, y) - img2.at<float>(x, y)) > 10e-2) {
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
  intptr_t sizesImage[2] = {inputImage.rows, inputImage.cols};
  intptr_t sizesKernel[2] = {kernelRows, kernelCols};

  // Define input, kernel, and output.
  MemRef<float, 2> input(inputImage, sizesImage);
  MemRef<float, 2> kernel(kernelArray, sizesKernel);
  MemRef<float, 2> output(sizesImage);

  for (int i = 0; i < inputImage.rows; i++)
    for (int j = 0; j < inputImage.cols; j++)
      output[i * inputImage.rows + j] = 0;

  Mat kernel1 = Mat(kernelRows, kernelCols, CV_32FC1, kernelArray);
  Mat opencvOutput;

  _mlir_ciface_corr_2d_constant_padding(&input, &kernel, &output, x, y, 0);

  filter2D(inputImage, opencvOutput, CV_32FC1, kernel1, cv::Point(x, y), 0.0,
           cv::BORDER_CONSTANT);

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
    {
      cv::Mat testImage = cv::imread(getTestImageName(), cv::IMREAD_GRAYSCALE);
      testKernel(testImage, get<1>(kernel.second), get<2>(kernel.second), get<0>(kernel.second));
    }
}

int main(int argc, char **argv)
{
  FilterTest::setImageNames(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
