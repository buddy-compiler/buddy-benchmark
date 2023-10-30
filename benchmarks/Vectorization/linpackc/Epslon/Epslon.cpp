//===- Epslon.cpp ---------------------------------------------------------===//
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
// This file provides the linpackc epslon function.
//
//===----------------------------------------------------------------------===//
#include <math.h>
#include <stdio.h>
/*----------------------*/
TYPE_PLACEHOLDER
epslon_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(TYPE_PLACEHOLDER x)
/*
     estimate unit roundoff in quantities of size x.
*/

{
  TYPE_PLACEHOLDER a, b, c, eps;
  a = 4.0e0 / 3.0e0;
  eps = TYPE_PLACEHOLDER(0.0);
  while (eps == TYPE_PLACEHOLDER(0.0)) {
    b = a - TYPE_PLACEHOLDER(1.0);
    c = b + b + b;
    eps = fabs((double)(c - TYPE_PLACEHOLDER(1.0)));
  }
  return (eps * fabs((double)x));
}
