//===- Daxpy.h ------------------------------------------------------------===//
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
// This file provides the linpackc daxpy function declarations.
//
//===----------------------------------------------------------------------===//
#include <math.h>
#include <stdio.h>

void daxpy_ROLL_float_gcc(int n, float da, float dx[], int incx, float dy[],
                          int incy);
void daxpy_ROLL_double_gcc(int n, double da, double dx[], int incx, double dy[],
                           int incy);
void daxpy_UNROLL_float_gcc(int n, float da, float dx[], int incx, float dy[],
                            int incy);
void daxpy_UNROLL_double_gcc(int n, double da, double dx[], int incx,
                             double dy[], int incy);
void daxpy_ROLL_float_clang(int n, float da, float dx[], int incx, float dy[],
                            int incy);
void daxpy_ROLL_double_clang(int n, double da, double dx[], int incx,
                             double dy[], int incy);
void daxpy_UNROLL_float_clang(int n, float da, float dx[], int incx, float dy[],
                              int incy);
void daxpy_UNROLL_double_clang(int n, double da, double dx[], int incx,
                               double dy[], int incy);
