#include <cstdio>
#include <cmath>

void daxpy_ROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(int n,TYPE_PLACEHOLDER da, TYPE_PLACEHOLDER dx[],int incx, TYPE_PLACEHOLDER dy[], int incy)
/*
     constant times a vector plus a vector.
     jack dongarra, linpack, 3/11/78.
*/
{
	int i,ix,iy,m,mp1;

	if(n <= 0) return;
	if (da == TYPE_PLACEHOLDER(0.0)) return;

	if(incx != 1 || incy != 1) {

		/* code for unequal increments or equal increments
		   not equal to 1 					*/

		ix = 0;
		iy = 0;
		if(incx < 0) ix = (-n+1)*incx;
		if(incy < 0)iy = (-n+1)*incy;
		for (i = 0;i < n; i++) {
			dy[iy] = dy[iy] + da*dx[ix];
			ix = ix + incx;
			iy = iy + incy;
		}
      		return;
	}
	/* code for both increments equal to 1 */
	for (i = 0;i < n; i++) {
		dy[i] = dy[i] + da*dx[i];
	}

}

void daxpy_UNROLL_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(int n,TYPE_PLACEHOLDER da, TYPE_PLACEHOLDER dx[],int incx,TYPE_PLACEHOLDER dy[], int incy){
	int i,ix,iy,m,mp1;

	if(n <= 0) return;
	if (da == TYPE_PLACEHOLDER(0.0)) return;

	if(incx != 1 || incy != 1) {

		/* code for unequal increments or equal increments
		   not equal to 1 					*/

		ix = 0;
		iy = 0;
		if(incx < 0) ix = (-n+1)*incx;
		if(incy < 0)iy = (-n+1)*incy;
		for (i = 0;i < n; i++) {
			dy[iy] = dy[iy] + da*dx[ix];
			ix = ix + incx;
			iy = iy + incy;
		}
      		return;
	}
	/* code for both increments equal to 1 */
	m = n % 4;
	if ( m != 0) {
		for (i = 0; i < m; i++) 
			dy[i] = dy[i] + da*dx[i];
		if (n < 4) return;
	}
	for (i = m; i < n; i = i + 4) {
		dy[i] = dy[i] + da*dx[i];
		dy[i+1] = dy[i+1] + da*dx[i+1];
		dy[i+2] = dy[i+2] + da*dx[i+2];
		dy[i+3] = dy[i+3] + da*dx[i+3];
	}
}