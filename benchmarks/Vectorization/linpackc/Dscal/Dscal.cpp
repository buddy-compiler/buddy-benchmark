//===- Dscal.cpp ----------------------------------------------------------===//
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
// This file provides the linpackc dscal function.
//
//===----------------------------------------------------------------------===//
void dscal_ROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(int n,
                                                      TYPE_PLACEHOLDER da,
                                                      TYPE_PLACEHOLDER dx[],
                                                      int incx)

/*     scales a vector by a constant.
      jack dongarra, linpack, 3/11/78.
*/
{
  int i, m, mp1, nincx;

  if (n <= 0)
    return;
  if (incx != 1) {

    /* code for increment not equal to 1 */

    nincx = n * incx;
    for (i = 0; i < nincx; i = i + incx)
      dx[i] = da * dx[i];
    return;
  }

  /* code for increment equal to 1 */
  for (i = 0; i < n; i++)
    dx[i] = da * dx[i];
}

void dscal_UNROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(int n,
                                                        TYPE_PLACEHOLDER da,
                                                        TYPE_PLACEHOLDER dx[],
                                                        int incx)

/*     scales a vector by a constant.
      jack dongarra, linpack, 3/11/78.
*/
{
  int i, m, mp1, nincx;

  if (n <= 0)
    return;
  if (incx != 1) {

    /* code for increment not equal to 1 */

    nincx = n * incx;
    for (i = 0; i < nincx; i = i + incx)
      dx[i] = da * dx[i];
    return;
  }

  /* code for increment equal to 1 */
  m = n % 5;
  if (m != 0) {
    for (i = 0; i < m; i++)
      dx[i] = da * dx[i];
    if (n < 5)
      return;
  }
  for (i = m; i < n; i = i + 5) {
    dx[i] = da * dx[i];
    dx[i + 1] = da * dx[i + 1];
    dx[i + 2] = da * dx[i + 2];
    dx[i + 3] = da * dx[i + 3];
    dx[i + 4] = da * dx[i + 4];
  }
}
