//===- Idamax.h -----------------------------------------------------------===//
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
// This file provides the linpackc idamax function declarations.
//
//===----------------------------------------------------------------------===//
#include <math.h>
#include <stdio.h>

int idamax_float_gcc(int n, float dx[], int incx);
int idamax_double_gcc(int n, double dx[], int incx);
int idamax_float_clang(int n, float dx[], int incx);
int idamax_double_clang(int n, double dx[], int incx);
