//===- Dmxpy.h ------------------------------------------------------------===//
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
// This file provides the linpackc dmxpy function declarations.
//
//===----------------------------------------------------------------------===//
#include <math.h>
#include <stdio.h>
void dmxpy_float_gcc(int n1, float y[], int n2, int ldm, float x[], float m[]);
void dmxpy_double_gcc(int n1, double y[], int n2, int ldm, double x[],
                      double m[]);
void dmxpy_float_clang(int n1, float y[], int n2, int ldm, float x[],
                       float m[]);
void dmxpy_double_clang(int n1, double y[], int n2, int ldm, double x[],
                        double m[]);
                        