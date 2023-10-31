//===- Dgefa.cpp ----------------------------------------------------------===//
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
// This file provides the linpackc dgefa function.
//
//===----------------------------------------------------------------------===//
#include <Daxpy/Daxpy.h>
#include <Dscal/Dscal.h>
#include <Idamax/Idamax.h>
#include <math.h>
#include <stdio.h>

void dgefa_ROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(TYPE_PLACEHOLDER a[],
                                                      int lda, int n,
                                                      int ipvt[], int *info)
/* We would like to declare a[][lda], but c does not allow it.  In this
function, references to a[i][j] are written a[lda*i+j].  */
/*
     dgefa factors a double precision matrix by gaussian elimination.

     dgefa is usually called by dgeco, but it can be called
     directly with a saving in time if  rcond  is not needed.
     (time for dgeco) = (1 + 9/n)*(time for dgefa) .

     on entry

        a       TYPE_PLACEHOLDER precision[n][lda]
                the matrix to be factored.

        lda     integer
                the leading dimension of the array  a .

        n       integer
                the order of the matrix  a .

     on return

        a       an upper triangular matrix and the multipliers
                which were used to obtain it.
                the factorization can be written  a = l*u  where
                l  is a product of permutation and unit lower
                triangular matrices and  u  is upper triangular.

        ipvt    integer[n]
                an integer vector of pivot indices.

        info    integer
                = 0  normal value.
                = k  if  u[k][k] .eq. 0.0 .  this is not an error
                     condition for this subroutine, but it does
                     indicate that dgesl or dgedi will divide by zero
                     if called.  use  rcond  in dgeco for a reliable
                     indication of singularity.

     linpack. this version dated 08/14/78 .
     cleve moler, university of new mexico, argonne national lab.

     functions

     blas daxpy,dscal,idamax
*/

{
  /*     internal variables	*/

  TYPE_PLACEHOLDER t;
  int j, k, kp1, l, nm1;

  /*     gaussian elimination with partial pivoting	*/

  *info = 0;
  nm1 = n - 1;
  if (nm1 >= 0) {
    for (k = 0; k < nm1; k++) {
      kp1 = k + 1;

      /* find l = pivot index	*/

      l = idamax_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(n - k, &a[lda * k + k],
                                                       1) +
          k;
      ipvt[k] = l;

      /* zero pivot implies this column already
         triangularized */

      if (a[lda * k + l] != TYPE_PLACEHOLDER(0.0)) {

        /* interchange if necessary */

        if (l != k) {
          t = a[lda * k + l];
          a[lda * k + l] = a[lda * k + k];
          a[lda * k + k] = t;
        }

        /* compute multipliers */

        t = -TYPE_PLACEHOLDER(1.0) / a[lda * k + k];
        dscal_ROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(
            n - (k + 1), t, &a[lda * k + k + 1], 1);

        /* row elimination with column indexing */

        for (j = kp1; j < n; j++) {
          t = a[lda * j + l];
          if (l != k) {
            a[lda * j + l] = a[lda * j + k];
            a[lda * j + k] = t;
          }
          daxpy_ROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(
              n - (k + 1), t, &a[lda * k + k + 1], 1, &a[lda * j + k + 1], 1);
        }
      } else {
        *info = k;
      }
    }
  }
  ipvt[n - 1] = n - 1;
  if (a[lda * (n - 1) + (n - 1)] == TYPE_PLACEHOLDER(0.0))
    *info = n - 1;
}

void dgefa_UNROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(TYPE_PLACEHOLDER a[],
                                                        int lda, int n,
                                                        int ipvt[], int *info)
/* We would like to declare a[][lda], but c does not allow it.  In this
function, references to a[i][j] are written a[lda*i+j].  */
/*
     dgefa factors a double precision matrix by gaussian elimination.

     dgefa is usually called by dgeco, but it can be called
     directly with a saving in time if  rcond  is not needed.
     (time for dgeco) = (1 + 9/n)*(time for dgefa) .

     on entry

        a       TYPE_PLACEHOLDER precision[n][lda]
                the matrix to be factored.

        lda     integer
                the leading dimension of the array  a .

        n       integer
                the order of the matrix  a .

     on return

        a       an upper triangular matrix and the multipliers
                which were used to obtain it.
                the factorization can be written  a = l*u  where
                l  is a product of permutation and unit lower
                triangular matrices and  u  is upper triangular.

        ipvt    integer[n]
                an integer vector of pivot indices.

        info    integer
                = 0  normal value.
                = k  if  u[k][k] .eq. 0.0 .  this is not an error
                     condition for this subroutine, but it does
                     indicate that dgesl or dgedi will divide by zero
                     if called.  use  rcond  in dgeco for a reliable
                     indication of singularity.

     linpack. this version dated 08/14/78 .
     cleve moler, university of new mexico, argonne national lab.

     functions

     blas daxpy,dscal,idamax
*/

{
  /*     internal variables	*/

  TYPE_PLACEHOLDER t;
  int j, k, kp1, l, nm1;

  /*     gaussian elimination with partial pivoting	*/

  *info = 0;
  nm1 = n - 1;
  if (nm1 >= 0) {
    for (k = 0; k < nm1; k++) {
      kp1 = k + 1;

      /* find l = pivot index	*/

      l = idamax_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(n - k, &a[lda * k + k],
                                                       1) +
          k;
      ipvt[k] = l;

      /* zero pivot implies this column already
         triangularized */

      if (a[lda * k + l] != TYPE_PLACEHOLDER(0.0)) {

        /* interchange if necessary */

        if (l != k) {
          t = a[lda * k + l];
          a[lda * k + l] = a[lda * k + k];
          a[lda * k + k] = t;
        }

        /* compute multipliers */

        t = -TYPE_PLACEHOLDER(1.0) / a[lda * k + k];
        dscal_UNROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(
            n - (k + 1), t, &a[lda * k + k + 1], 1);

        /* row elimination with column indexing */

        for (j = kp1; j < n; j++) {
          t = a[lda * j + l];
          if (l != k) {
            a[lda * j + l] = a[lda * j + k];
            a[lda * j + k] = t;
          }
          daxpy_UNROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(
              n - (k + 1), t, &a[lda * k + k + 1], 1, &a[lda * j + k + 1], 1);
        }
      } else {
        *info = k;
      }
    }
  }
  ipvt[n - 1] = n - 1;
  if (a[lda * (n - 1) + (n - 1)] == TYPE_PLACEHOLDER(0.0))
    *info = n - 1;
}
