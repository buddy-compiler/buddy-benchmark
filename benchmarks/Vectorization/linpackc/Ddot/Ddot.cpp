//===- Ddot.cpp -----------------------------------------------------------===//
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
// This file provides the linpackc ddot function.
//
//===----------------------------------------------------------------------===//
TYPE_PLACEHOLDER ddot_ROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(
    int n, TYPE_PLACEHOLDER dx[], int incx, TYPE_PLACEHOLDER dy[], int incy)
/*
     forms the dot product of two vectors.
     jack dongarra, linpack, 3/11/78.
*/
{
  TYPE_PLACEHOLDER dtemp;
  int i, ix, iy, m, mp1;

  dtemp = TYPE_PLACEHOLDER(0.0);

  if (n <= 0)
    return (TYPE_PLACEHOLDER(0.0));

  if (incx != 1 || incy != 1) {

    /* code for unequal increments or equal increments
       not equal to 1					*/

    ix = 0;
    iy = 0;
    if (incx < 0)
      ix = (-n + 1) * incx;
    if (incy < 0)
      iy = (-n + 1) * incy;
    for (i = 0; i < n; i++) {
      dtemp = dtemp + dx[ix] * dy[iy];
      ix = ix + incx;
      iy = iy + incy;
    }
    return (dtemp);
  }

  /* code for both increments equal to 1 */
  for (i = 0; i < n; i++)
    dtemp = dtemp + dx[i] * dy[i];
  return (dtemp);
}

TYPE_PLACEHOLDER ddot_UNROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(
    int n, TYPE_PLACEHOLDER dx[], int incx, TYPE_PLACEHOLDER dy[], int incy)
/*
     forms the dot product of two vectors.
     jack dongarra, linpack, 3/11/78.
*/
{
  TYPE_PLACEHOLDER dtemp;
  int i, ix, iy, m, mp1;

  dtemp = TYPE_PLACEHOLDER(0.0);

  if (n <= 0)
    return (TYPE_PLACEHOLDER(0.0));

  if (incx != 1 || incy != 1) {

    /* code for unequal increments or equal increments
       not equal to 1					*/

    ix = 0;
    iy = 0;
    if (incx < 0)
      ix = (-n + 1) * incx;
    if (incy < 0)
      iy = (-n + 1) * incy;
    for (i = 0; i < n; i++) {
      dtemp = dtemp + dx[ix] * dy[iy];
      ix = ix + incx;
      iy = iy + incy;
    }
    return (dtemp);
  }
  m = n % 5;
  if (m != 0) {
    for (i = 0; i < m; i++)
      dtemp = dtemp + dx[i] * dy[i];
    if (n < 5)
      return (dtemp);
  }
  for (i = m; i < n; i = i + 5) {
    dtemp = dtemp + dx[i] * dy[i] + dx[i + 1] * dy[i + 1] +
            dx[i + 2] * dy[i + 2] + dx[i + 3] * dy[i + 3] +
            dx[i + 4] * dy[i + 4];
  }
  return (dtemp);
}
