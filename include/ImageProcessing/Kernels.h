//===- Kernels.h ----------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the kernels for the image processing benchmarks.
//
//===----------------------------------------------------------------------===//

#ifndef IMAGE_PROCESSING_KERNELS
#define IMAGE_PROCESSING_KERNELS

// clang-format off
#include <vector>
#include <string>
#include <map>

static float prewittKernelAlign[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
static int prewittKernelRows = 3;
static int prewittKernelCols = 3;

static float sobel3x3KernelAlign[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
static int sobel3x3KernelRows = 3;
static int sobel3x3KernelCols = 3;

static float sobel5x5KernelAlign[25] = {2, 1, 0, -1, -2,
                                        3, 2, 0, -2, -3,
                                        4, 3, 0, -3, -4,
                                        3, 2, 0, -2, -3,
                                        2, 1, 0, -1, -2};
static int sobel5x5KernelRows = 5;
static int sobel5x5KernelCols = 5;

static float sobel7x7KernelAlign[49] = {3, 2, 1, 0, -1, -2, -3,
                                        4, 3, 2, 0, -2, -3, -4,
                                        5, 4, 3, 0, -3, -4, -5,
                                        6, 5, 4, 0, -4, -5, -6,
                                        5, 4, 3, 0, -3, -4, -5,
                                        4, 3, 2, 0, -2, -3, -4,
                                        3, 2, 1, 0, -1, -2, -3};
static int sobel7x7KernelRows = 7;
static int sobel7x7KernelCols = 7;

static float sobel9x9KernelAlign[81] = {4, 3, 2, 1, 0, -1, -2, -3, -4,
                                        5, 4, 3, 2, 0, -2, -3, -4, -5,
                                        6, 5, 4, 3, 0, -3, -4, -5, -6,
                                        7, 6, 5, 4, 0, -4, -5, -6, -7,
                                        8, 7, 6, 5, 0, -5, -6, -7, -8,
                                        7, 6, 5, 4, 0, -4, -5, -6, -7,
                                        6, 5, 4, 3, 0, -3, -4, -5, -6,
                                        5, 4, 3, 2, 0, -2, -3, -4, -5,
                                        4, 3, 2, 1, 0, -1, -2, -3, -4};
static int sobel9x9KernelRows = 9;
static int sobel9x9KernelCols = 9;

static float laplacianKernelAlign[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
static int laplacianKernelRows = 3;
static int laplacianKernelCols = 3;

static float logKernelAlign[25] = {0, 0, 1, 0, 0, 
                                   0, 1, 2, 1, 0,
                                   1, 2, -16, 2, 1,
                                   0, 1, 2, 1, 0,
                                   0, 0, 1, 0, 0};
static int logKernelRows = 5;
static int logKernelCols = 5;

static float random3x3KernelAlign[9] = {0, 4, 1, 
                                        3, 3, 8, 
                                        9, 6, 3};
static int random3x3KernelRows = 3;
static int random3x3KernelCols = 3;

static float random5x5KernelAlign[25] = {1, 9, 8, 8, 1, 
                                         6, 6, 3, 3, 7, 
                                         1, 9, 5, 5, 7, 
                                         4, 5, 4, 3, 2, 
                                         9, 8, 3, 8, 9};
static int random5x5KernelRows = 5;
static int random5x5KernelCols = 5;

static float random7x7KernelAlign[49] = {6, 3, 1, 5, 1, 9, 4, 
                                        0, 0, 8, 0, 5, 0, 2, 
                                        9, 1, 7, 6, 1, 3, 8, 
                                        4, 5, 2, 6, 3, 9, 1, 
                                        7, 7, 2, 6, 7, 8, 6, 
                                        2, 8, 9, 2, 2, 7, 8, 
                                        8, 9, 2, 3, 5, 3, 5};
static int random7x7KernelRows = 7;
static int random7x7KernelCols = 7;

static float random9x9KernelAlign[81] = {7, 9, 5, 0, 9, 8, 4, 9, 9, 
                                         4, 7, 0, 8, 3, 3, 2, 8, 7, 
                                         1, 0, 5, 5, 9, 4, 1, 7, 0, 
                                         6, 1, 5, 4, 1, 8, 4, 8, 4, 
                                         4, 9, 5, 7, 5, 6, 7, 1, 3, 
                                         7, 2, 9, 8, 1, 4, 5, 7, 7, 
                                         7, 3, 4, 4, 3, 3, 4, 0, 7, 
                                         2, 7, 3, 5, 2, 3, 4, 2, 3, 
                                         9, 5, 2, 7, 7, 4, 8, 7, 2};
static int random9x9KernelRows = 9;
static int random9x9KernelCols = 9;

static float random11x11KernelAlign[121] = {4, 5, 7, 6, 3, 6, 6, 6, 1, 9, 8,
                                            5, 2, 0, 0, 9, 0, 1, 9, 7, 0, 8,
                                            7, 8, 4, 4, 1, 5, 5, 3, 4, 9, 8,
                                            4, 7, 1, 2, 3, 9, 3, 2, 9, 0, 7,
                                            9, 2, 6, 0, 5, 7, 9, 6, 5, 8, 6,
                                            0, 2, 7, 7, 7, 0, 3, 8, 0, 7, 5,
                                            4, 9, 0, 5, 4, 5, 5, 7, 2, 6, 1,
                                            0, 6, 7, 9, 7, 5, 6, 6, 1, 8, 0,
                                            0, 7, 8, 3, 3, 8, 3, 2, 6, 9, 2,
                                            6, 5, 8, 1, 0, 7, 5, 8, 9, 7, 5,
                                            6, 8, 4, 3, 7, 0, 6, 5, 1, 6, 5};
static int random11x11KernelRows = 11;
static int random11x11KernelCols = 11;

static float random13x13KernelAlign[169] = {8, 0, 6, 6, 7, 3, 4, 0, 6, 1, 5, 4, 3,
                                            5, 8, 4, 7, 8, 7, 0, 7, 9, 1, 9, 4, 6,
                                            5, 6, 0, 7, 6, 8, 9, 4, 7, 7, 7, 1, 9,
                                            3, 4, 7, 8, 8, 4, 8, 4, 2, 6, 3, 4, 3,
                                            5, 6, 5, 9, 4, 2, 5, 4, 9, 3, 4, 8, 7,
                                            1, 7, 4, 4, 9, 0, 9, 6, 0, 9, 2, 8, 3,
                                            4, 7, 8, 1, 2, 3, 7, 9, 4, 3, 1, 9, 7,
                                            2, 4, 3, 3, 3, 5, 0, 0, 9, 1, 0, 0, 9,
                                            0, 9, 2, 0, 4, 8, 9, 3, 9, 2, 8, 8, 1,
                                            3, 1, 5, 4, 0, 7, 9, 4, 0, 2, 9, 3, 2,
                                            0, 4, 4, 1, 4, 6, 2, 8, 9, 7, 6, 8, 2,
                                            8, 2, 0, 6, 6, 3, 0, 1, 8, 0, 8, 9, 6,
                                            9, 1, 5, 2, 6, 8, 8, 2, 1, 4, 9, 3, 2};
static int random13x13KernelRows = 13;
static int random13x13KernelCols = 13;

static float random15x15KernelAlign[225] = {0, 7, 6, 8, 5, 9, 6, 1, 6, 0, 7, 5, 6, 1, 6,
                                            2, 9, 8, 4, 0, 5, 5, 1, 8, 4, 0, 3, 0, 1, 1,
                                            5, 4, 9, 1, 2, 4, 2, 0, 8, 9, 0, 7, 6, 6, 8,
                                            3, 0, 7, 3, 4, 9, 8, 2, 0, 9, 6, 2, 2, 8, 6,
                                            4, 3, 0, 5, 6, 2, 9, 9, 4, 9, 0, 4, 6, 6, 2,
                                            6, 9, 3, 6, 4, 9, 5, 5, 1, 6, 4, 0, 0, 8, 8,
                                            6, 4, 4, 8, 9, 0, 2, 1, 1, 6, 0, 3, 0, 7, 0,
                                            5, 3, 1, 0, 1, 6, 9, 7, 3, 3, 5, 7, 3, 7, 5,
                                            3, 4, 0, 9, 2, 1, 2, 5, 2, 3, 3, 3, 7, 4, 2,
                                            9, 1, 5, 2, 1, 9, 0, 2, 6, 3, 7, 3, 0, 2, 0,
                                            8, 8, 6, 0, 7, 1, 1, 9, 6, 4, 5, 9, 7, 4, 5,
                                            9, 5, 8, 6, 7, 1, 5, 8, 6, 3, 1, 3, 6, 4, 8,
                                            9, 4, 6, 5, 4, 3, 6, 7, 5, 4, 1, 0, 6, 0, 6,
                                            3, 1, 1, 2, 0, 8, 5, 7, 6, 1, 1, 0, 7, 9, 6,
                                            5, 0, 2, 1, 6, 8, 6, 4, 5, 3, 1, 9, 3, 7, 9};
static int random15x15KernelRows = 15;
static int random15x15KernelCols = 15;

static std::map<std::string, std::tuple<float*, int, int>> kernelMap = {
    {"prewittKernelAlign", {prewittKernelAlign, prewittKernelRows, prewittKernelCols}},
    {"sobel3x3KernelAlign", {sobel3x3KernelAlign, sobel3x3KernelRows, sobel3x3KernelCols}},
    {"sobel5x5KernelAlign", {sobel5x5KernelAlign, sobel5x5KernelRows, sobel5x5KernelCols}},
    {"sobel7x7KernelAlign", {sobel7x7KernelAlign, sobel7x7KernelRows, sobel7x7KernelCols}},
    {"sobel9x9KernelAlign", {sobel9x9KernelAlign, sobel9x9KernelRows, sobel9x9KernelCols}},
    {"laplacianKernelAlign", {laplacianKernelAlign, laplacianKernelRows, laplacianKernelCols}},
    {"logKernelAlign", {logKernelAlign, logKernelRows, logKernelCols}},
    {"random3x3KernelAlign", {random3x3KernelAlign, random3x3KernelRows, random3x3KernelCols}},
    {"random5x5KernelAlign", {random5x5KernelAlign, random5x5KernelRows, random5x5KernelCols}},
    {"random7x7KernelAlign", {random7x7KernelAlign, random7x7KernelRows, random7x7KernelCols}},
    {"random9x9KernelAlign", {random9x9KernelAlign, random9x9KernelRows, random9x9KernelCols}},
    {"random11x11KernelAlign", {random11x11KernelAlign, random11x11KernelRows, random11x11KernelCols}},
    {"random13x13KernelAlign", {random13x13KernelAlign, random13x13KernelRows, random13x13KernelCols}},
    {"random15x15KernelAlign", {random15x15KernelAlign, random15x15KernelRows, random15x15KernelCols}}
};

static std::vector<std::string> imageNames = {
    "test_6x6.png", "YuTu.png",    "YuTu4.png",   "YuTu6.png",
    "YuTu8.png",    "YuTu16.png",  "YuTu32.png",  "YuTu64.png",
    "YuTu128.png",  "YuTu256.png", "YuTu512.png", "YuTu1024.png"};

// clang-format on

#endif // IMAGE_PROCESSING_KERNELS
