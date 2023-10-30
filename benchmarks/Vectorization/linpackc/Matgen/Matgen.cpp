//===- Matgen.cpp ---------------------------------------------------------===//
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
// This file provides the linpackc matgen function.
//
//===----------------------------------------------------------------------===//
void matgen_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(TYPE_PLACEHOLDER a[], int lda,
                                                  int n, TYPE_PLACEHOLDER b[],
                                                  TYPE_PLACEHOLDER *norma)

/* We would like to declare a[][lda], but c does not allow it.  In this
function, references to a[i][j] are written a[lda*j+i].  */

{
  int init, i, j;

  init = 1325;
  *norma = 0.0;
  for (j = 0; j < n; j++) {
    for (i = 0; i < n; i++) {
      init = 3125 * init % 65536;
      a[lda * j + i] = (init - 32768.0) / 16384.0;
      *norma = (a[lda * j + i] > *norma) ? a[lda * j + i] : *norma;
    }
  }
  for (i = 0; i < n; i++) {
    b[i] = 0.0;
  }
  for (j = 0; j < n; j++) {
    for (i = 0; i < n; i++) {
      b[i] = b[i] + a[lda * j + i];
    }
  }
}
