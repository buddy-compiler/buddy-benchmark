//===- Idamax.cpp ---------------------------------------------------------===//
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
// This file provides the linpackc idamax function.
//
//===----------------------------------------------------------------------===//
#include <math.h>
#include <stdio.h>
int idamax_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(int n, TYPE_PLACEHOLDER dx[],
                                                 int incx)

/*
     finds the index of element having max. absolute value.
     jack dongarra, linpack, 3/11/78.
*/
{
  TYPE_PLACEHOLDER dmax;
  int i, ix, itemp;

  if (n < 1)
    return (-1);
  if (n == 1)
    return (0);
  if (incx != 1) {

    /* code for increment not equal to 1 */

    ix = 0;
    dmax = fabs((double)dx[0]);
    ix = ix + incx;
    for (i = 1; i < n; i++) {
      if (fabs((double)dx[ix]) > dmax) {
        itemp = i;
        dmax = fabs((double)dx[ix]);
      }
      ix = ix + incx;
    }
  } else {

    /* code for increment equal to 1 */

    itemp = 0;
    dmax = fabs((double)dx[0]);
    for (i = 1; i < n; i++) {
      if (fabs((double)dx[i]) > dmax) {
        itemp = i;
        dmax = fabs((double)dx[i]);
      }
    }
  }
  return (itemp);
}
