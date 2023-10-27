#include <stdio.h>
#include <math.h>
/*----------------------*/ 
TYPE_PLACEHOLDER epslon_TYPE_PLACEHOLDER_COMPILER_PLACEHOLDER(TYPE_PLACEHOLDER x)
/*
     estimate unit roundoff in quantities of size x.
*/

{
	TYPE_PLACEHOLDER a,b,c,eps;
	a = 4.0e0/3.0e0;
	eps = TYPE_PLACEHOLDER(0.0);
	while (eps == TYPE_PLACEHOLDER(0.0)) {
		b = a - TYPE_PLACEHOLDER(1.0);
		c = b + b + b;
		eps = fabs((double)(c-TYPE_PLACEHOLDER(1.0)));
	}
	return(eps*fabs((double)x));
}
